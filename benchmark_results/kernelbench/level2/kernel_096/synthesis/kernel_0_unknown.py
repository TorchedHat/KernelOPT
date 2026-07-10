import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C_in, C_out,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    # X strides: (N, C_in, ID, IH, IW)
    sx_n, sx_c, sx_d, sx_h, sx_w,
    # W strides: (C_in, C_out, KD, KH, KW)
    sw_ci, sw_co, sw_kd, sw_kh, sw_kw,
    # Y strides: (N, C_out, OD, OH, OW)
    sy_n, sy_co, sy_od, sy_oh, sy_ow,
    M, K,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    M = N * OD * OH * OW  (output positions)
    N_dim = C_out
    K = C_in * KD * KH * KW
    
    Y[n, co, od, oh, ow] = sum over (ci, kd, kh, kw) where:
        (od + pad_d - kd) % stride_d == 0 and id = (od + pad_d - kd) / stride_d in [0, ID)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (n, od, oh, ow)
    OHW = OH * OW
    ODHW = OD * OH * OW
    
    n_m = offs_m // ODHW
    rem = offs_m % ODHW
    od_m = rem // OHW
    rem2 = rem % OHW
    oh_m = rem2 // OW
    ow_m = rem2 % OW
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Decode K -> (ci, kd, kh, kw)
        KHW = KH * KW
        KDHW = KD * KH * KW
        ci_k = offs_k // KDHW
        rem_k = offs_k % KDHW
        kd_k = rem_k // KHW
        rem_k2 = rem_k % KHW
        kh_k = rem_k2 // KW
        kw_k = rem_k2 % KW
        
        # For ConvTranspose3d:
        # id = (od + pad_d - kd) / stride_d  (must be integer and in bounds)
        # Compute numerator
        od_plus_pad_d = od_m[:, None] + pad_d - kd_k[None, :]  # (BLOCK_M, BLOCK_K)
        oh_plus_pad_h = oh_m[:, None] + pad_h - kh_k[None, :]
        ow_plus_pad_w = ow_m[:, None] + pad_w - kw_k[None, :]
        
        # Check divisibility and bounds
        valid_d = (od_plus_pad_d % stride_d == 0)
        valid_h = (oh_plus_pad_h % stride_h == 0)
        valid_w = (ow_plus_pad_w % stride_w == 0)
        
        id_in = od_plus_pad_d // stride_d
        ih_in = oh_plus_pad_h // stride_h
        iw_in = ow_plus_pad_w // stride_w
        
        in_bounds = (
            (id_in >= 0) & (id_in < ID) &
            (ih_in >= 0) & (ih_in < IH) &
            (iw_in >= 0) & (iw_in < IW) &
            (n_m[:, None] < N) &
            k_mask[None, :]
        )
        
        # X[n, ci, id, ih, iw]
        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=in_bounds, other=0.0)
        
        # W[ci, co, kd, kh, kw] -> shape (BLOCK_K, BLOCK_N)
        w_ptrs = (
        )
        w_mask = k_mask[:, None] & (offs_n[None, :] < C_out)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias and scale
    bias = tl.load(B_ptr + offs_n, mask=offs_n < C_out, other=0.0)
    acc = (acc + bias[None, :]) * scale
    
    # Store Y[n, co, od, oh, ow]
    y_ptrs = (
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_out'],
)
@triton.jit
def maxpool3d_kernel(
    X_ptr, Y_ptr,
    N, C,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    # X strides
    sx_n, sx_c, sx_d, sx_h, sx_w,
    # Y strides
    sy_n, sy_c, sy_d, sy_h, sy_w,
    total_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_out
    
    # Decode: total_out = N * C * OD * OH * OW
    OHW = OH * OW
    ODHW = OD * OH * OW
    CODHW = C * ODHW
    
    n_idx = offs // CODHW
    rem = offs % CODHW
    c_idx = rem // ODHW
    rem2 = rem % ODHW
    od_idx = rem2 // OHW
    rem3 = rem2 % OHW
    oh_idx = rem3 // OW
    ow_idx = rem3 % OW
    
    # Max pool over KD x KH x KW window
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                id_in = od_idx * KD + kd
                ih_in = oh_idx * KH + kh
                iw_in = ow_idx * KW + kw
                
                in_bounds = (
                    (id_in < ID) & (ih_in < IH) & (iw_in < IW) &
                    (n_idx < N) & (c_idx < C)
                )
                
                x_ptrs = (
                )
                x_val = tl.load(x_ptrs, mask=in_bounds, other=float('-inf'))
                max_val = tl.maximum(max_val, x_val)
    
    y_ptrs = (
    )
    tl.store(y_ptrs, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 32, 'BLOCK_SPATIAL': 256}, num_warps=4),
        triton.Config({'BLOCK_NC': 64, 'BLOCK_SPATIAL': 128}, num_warps=4),
        triton.Config({'BLOCK_NC': 16, 'BLOCK_SPATIAL': 512}, num_warps=8),
        triton.Config({'BLOCK_NC': 128, 'BLOCK_SPATIAL': 64}, num_warps=8),
    ],
    key=['N', 'C', 'spatial'],
)
@triton.jit
def global_avg_pool_clamp_kernel(
    X_ptr, Y_ptr,
    N, C,
    ID, IH, IW,
    spatial,
    clamp_min, clamp_max,
    # X strides
    sx_n, sx_c, sx_d, sx_h, sx_w,
    BLOCK_NC: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    nc_idx = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    
    n_idx = nc_idx // C
    c_idx = nc_idx % C
    
    nc_mask = nc_idx < (N * C)
    
    acc = tl.zeros((BLOCK_NC,), dtype=tl.float32)
    
    for sp_start in range(0, spatial, BLOCK_SPATIAL):
        sp_offs = sp_start + tl.arange(0, BLOCK_SPATIAL)
        sp_mask = sp_offs < spatial
        
        # Decode spatial -> (d, h, w)
        IHW = IH * IW
        d_idx = sp_offs // IHW
        rem = sp_offs % IHW
        h_idx = rem // IW
        w_idx = rem % IW
        
        # Load X[n, c, d, h, w] for each (nc, spatial) pair
        # Shape: (BLOCK_NC, BLOCK_SPATIAL)
        x_ptrs = (
        )
        combined_mask = nc_mask[:, None] & sp_mask[None, :]
        x_val = tl.load(x_ptrs, mask=combined_mask, other=0.0)
        acc += tl.sum(x_val, axis=1)
    
    result = acc / spatial
    # Clamp
    result = tl.maximum(result, clamp_min)
    result = tl.minimum(result, clamp_max)
    
    # Store Y[n, c, 0, 0, 0]
    y_ptrs = Y_ptr + n_idx * C + c_idx
    tl.store(y_ptrs, result, mask=nc_mask)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, x):
    # x: (N, C_in, ID, IH, IW) = (128, 3, 16, 32, 32)
    # w: (C_in, C_out, KD, KH, KW) = (3, 16, 3, 3, 3)
    N, C_in, ID, IH, IW = x.shape
    C_in_w, C_out, KD, KH, KW = p_conv_transpose_weight.shape
    
    stride_d, stride_h, stride_w = 2, 2, 2
    pad_d, pad_h, pad_w = 1, 1, 1
    scale = 0.5
    
    # ConvTranspose3d output size:
    # OD = (ID - 1) * stride_d - 2 * pad_d + KD = (16-1)*2 - 2 + 3 = 31
    OD = (ID - 1) * stride_d - 2 * pad_d + KD
    OH = (IH - 1) * stride_h - 2 * pad_h + KH
    OW = (IW - 1) * stride_w - 2 * pad_w + KW
    
    # Allocate conv output
    conv_out = torch.empty((N, C_out, OD, OH, OW), device=x.device, dtype=torch.float32)
    
    M = N * OD * OH * OW
    K = C_in * KD * KH * KW
    
    grid_conv = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    
    conv_transpose3d_kernel[grid_conv](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        N, C_in, C_out,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        # X strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        # W strides
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        p_conv_transpose_weight.stride(4),
        # Y strides
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
        M, K,
        scale,
    )
    
    # MaxPool3d with kernel=2, stride=2 (default stride=kernel_size)
    mp_k = 2
    mp_OD = OD // mp_k
    mp_OH = OH // mp_k
    mp_OW = OW // mp_k
    
    pool_out = torch.empty((N, C_out, mp_OD, mp_OH, mp_OW), device=x.device, dtype=torch.float32)
    total_pool = N * C_out * mp_OD * mp_OH * mp_OW
    
    grid_pool = lambda meta: (triton.cdiv(total_pool, meta['BLOCK_SIZE']),)
    
    maxpool3d_kernel[grid_pool](
        conv_out, pool_out,
        N, C_out,
        OD, OH, OW,
        mp_OD, mp_OH, mp_OW,
        mp_k, mp_k, mp_k,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
        pool_out.stride(0), pool_out.stride(1), pool_out.stride(2),
        pool_out.stride(3), pool_out.stride(4),
        total_pool,
    )
    
    # Global average pool + clamp -> (N, C_out, 1, 1, 1)
    final_out = torch.empty((N * C_out,), device=x.device, dtype=torch.float32)
    spatial = mp_OD * mp_OH * mp_OW
    
    grid_gap = lambda meta: (triton.cdiv(N * C_out, meta['BLOCK_NC']),)
    
    global_avg_pool_clamp_kernel[grid_gap](
        pool_out, final_out,
        N, C_out,
        mp_OD, mp_OH, mp_OW,
        spatial,
        0.0, 1.0,
        pool_out.stride(0), pool_out.stride(1), pool_out.stride(2),
        pool_out.stride(3), pool_out.stride(4),
    )
    
    return final_out.reshape(N, C_out, 1, 1, 1)


def get_inputs():
    torch.manual_seed(42)
    p_conv_transpose_weight = torch.randn(3, 16, 3, 3, 3)
    p_conv_transpose_bias = torch.randn(16)
    x = torch.rand(128, 3, 16, 32, 32)
    return [p_conv_transpose_weight, p_conv_transpose_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]
    
    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        w, b, x = inputs
        
        # Reference computation
        conv_t = nn.ConvTranspose3d(3, 16, 3, stride=2, padding=1)
        conv_t.weight.data = w
        conv_t.bias.data = b
        
        x_ref = x.cuda()
        with torch.no_grad():
            ref = conv_t(x_ref)
            ref = ref * 0.5
            ref = nn.MaxPool3d(kernel_size=2)(ref)
            ref = nn.AdaptiveAvgPool3d((1, 1, 1))(ref)
            ref = torch.clamp(ref, 0, 1)
        
        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Ref range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")
        
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6f}")
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")