import torch
import triton
import triton.language as tl

# ConvTranspose3d kernel using scatter approach
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M, N_out, K,
    s_xn, s_xc, s_xd, s_xh, s_xw,
    s_wic, s_woc, s_wkd, s_wkh, s_wkw,
    s_yn, s_yc, s_yd, s_yh, s_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M: each M corresponds to (batch, out_d, out_h, out_w)
    # M = N * OUT_D * OUT_H * OUT_W
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * KD * KH * KW
    # For ConvTranspose3d: output[n, oc, od, oh, ow] += input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where od = id*STRIDE_D - PAD_D + kd, etc.
    # So for given output position (od, oh, ow), we need:
    # id = (od + PAD_D - kd) / STRIDE_D (must be integer and in range)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K: K = IN_C * KD * KH * KW
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        ic_idx = tmp_k // KD
        
        # For ConvTranspose3d: input position
        # od = id * STRIDE_D - PAD_D + kd => id = (od + PAD_D - kd) / STRIDE_D
        od_val = out_d[:, None]  # (BLOCK_M, 1)
        oh_val = out_h[:, None]
        ow_val = out_w[:, None]
        
        kd_val = kd_idx[None, :]  # (1, BLOCK_K)
        kh_val = kh_idx[None, :]
        kw_val = kw_idx[None, :]
        ic_val = ic_idx[None, :]
        
        id_num = od_val + PAD_D - kd_val
        ih_num = oh_val + PAD_H - kh_val
        iw_num = ow_val + PAD_W - kw_val
        
        id_valid = (id_num % STRIDE_D) == 0
        ih_valid = (ih_num % STRIDE_H) == 0
        iw_valid = (iw_num % STRIDE_W) == 0
        
        id_idx = id_num // STRIDE_D
        ih_idx = ih_num // STRIDE_H
        iw_idx = iw_num // STRIDE_W
        
        in_bounds = (id_idx >= 0) & (id_idx < IN_D) & \
                    (ih_idx >= 0) & (ih_idx < IN_H) & \
                    (iw_idx >= 0) & (iw_idx < IN_W) & \
                    id_valid & ih_valid & iw_valid & \
                    (offs_k[None, :] < K) & \
                    (offs_m[:, None] < M) & \
                    (ic_val < IN_C)
        
        x_ptrs = X_ptr + batch[:, None] * s_xn + ic_val * s_xc + id_idx * s_xd + ih_idx * s_xh + iw_idx * s_xw
        x_tile = tl.load(x_ptrs, mask=in_bounds, other=0.0)
        
        # Weight: [IN_C, OUT_C, KD, KH, KW]
        w_ptrs = W_ptr + ic_val * s_wic + offs_n[None, :] * s_woc + kd_val * s_wkd + kh_val * s_wkh + kw_val * s_wkw
        # Need to broadcast properly
        # x_tile: (BLOCK_M, BLOCK_K), w_tile: (BLOCK_K, BLOCK_N)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_out)
        
        # Load weight as (BLOCK_K, BLOCK_N)
        w_ptrs2 = W_ptr + ic_idx[:, None] * s_wic + offs_n[None, :] * s_woc + kd_idx[:, None] * s_wkd + kh_idx[:, None] * s_wkh + kw_idx[:, None] * s_wkw
        w_tile = tl.load(w_ptrs2, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_out, other=0.0)
    acc = acc + bias[None, :]
    
    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_out)
    y_ptrs = Y_ptr + batch[:, None] * s_yn + offs_n[None, :] * s_yc + out_d[:, None] * s_yd + out_h[:, None] * s_yh + out_w[:, None] * s_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


# MaxPool3d + Softmax + Subtract + Swish + Max fused kernel
# Input: (N, C, D, H, W) after conv_transpose
# MaxPool3d: kernel=2, stride=2, padding=0 -> (N, C, D//2, H//2, W//2)
# Then softmax over C dim, subtract, swish, max over C
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16, 'BLOCK_SPATIAL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 16, 'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 16, 'BLOCK_SPATIAL': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_C': 16, 'BLOCK_SPATIAL': 256}, num_warps=8, num_stages=2),
    ],
    key=['C', 'SPATIAL'],
)
@triton.jit
def pool_softmax_swish_max_kernel(
    N, C,
    IN_D, IN_H, IN_W,
    OUT_D, OUT_H, OUT_W,
    POOL_KD, POOL_KH, POOL_KW,
    POOL_SD, POOL_SH, POOL_SW,
    s_xn, s_xc, s_xd, s_xh, s_xw,
    BLOCK_C: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # Each program handles one (batch, spatial_out) position
    # We iterate over all C channels to compute softmax, then subtract, swish, max
    pid = tl.program_id(0)
    
    # pid indexes into (N * SPATIAL)
    total = N * SPATIAL
    sp_idx = pid % SPATIAL
    n_idx = pid // SPATIAL
    
    if n_idx >= N:
        return
    
    # Decode spatial output index
    ow = sp_idx % OUT_W
    tmp = sp_idx // OUT_W
    oh = tmp % OUT_H
    od = tmp // OUT_H
    
    # For each channel, compute max pool over the pool window
    # Pool window: [od*POOL_SD : od*POOL_SD+POOL_KD, ...]
    id_start = od * POOL_SD
    ih_start = oh * POOL_SH
    iw_start = ow * POOL_SW
    
    # Load all C channel values after max pooling
    # We need to store them to compute softmax
    # Since C=16 fits in registers, we can do this
    
    # First pass: compute max pool for each channel and find global max for softmax
    c_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    
    # For each channel, compute max pool
    # Pool window iteration
    pool_max = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)
    
    for kd in range(POOL_KD):
        for kh in range(POOL_KH):
            for kw in range(POOL_KW):
                id_idx = id_start + kd
                ih_idx = ih_start + kh
                iw_idx = iw_start + kw
                
                valid = (id_idx < IN_D) & (ih_idx < IN_H) & (iw_idx < IN_W)
                
                x_ptrs = X_ptr + n_idx * s_xn + offs_c * s_xc + id_idx * s_xd + ih_idx * s_xh + iw_idx * s_xw
                x_vals = tl.load(x_ptrs, mask=c_mask & valid, other=float('-inf'))
                pool_max = tl.maximum(pool_max, x_vals)
    
    # pool_max now has max-pooled values for each channel
    # Compute softmax over channels
    sm_max = tl.max(pool_max, axis=0)
    exp_vals = tl.exp(pool_max - sm_max)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / exp_sum
    
    # Subtract parameter
    sub_vals = tl.load(SUB_ptr + offs_c, mask=c_mask, other=0.0)
    
    # Subtract
    after_sub = softmax_vals - sub_vals
    
    # Swish: sigmoid(x) * x
    swish_vals = tl.sigmoid(after_sub) * after_sub
    
    # Max over channels
    # Need to find both max value and index
    # Manual reduction since we need argmax
    max_val = tl.full((), float('-inf'), dtype=tl.float32)
    max_idx = tl.zeros((), dtype=tl.int64)
    
    # We'll do this by iterating - but with BLOCK_C=16 we can use vectorized approach
    # Use a trick: compute max value first, then find index
    final_max = tl.max(swish_vals, axis=0)
    
    # Find argmax: compare each element to max
    is_max = swish_vals == final_max
    # Get first index where is_max is true
    # Use: argmax = sum of (c * is_max) / sum(is_max) won't work for ties
    # Better: use masked indices
    c_indices = offs_c.to(tl.int64)
    # Set non-max to large value, find min index among max positions
    not_max_idx = tl.where(is_max, c_indices, tl.full((BLOCK_C,), 999999, dtype=tl.int64))
    argmax = tl.min(not_max_idx, axis=0)
    
    # Output index in (N, D_out, H_out, W_out)
    out_idx = n_idx * OUT_D * OUT_H * OUT_W + od * OUT_H * OUT_W + oh * OUT_W + ow
    
    tl.store(OUT_VAL_ptr + out_idx, final_max)
    tl.store(OUT_IDX_ptr + out_idx, argmax)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_subtract, x):
    # ConvTranspose3d parameters
    N, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = p_conv_transpose_weight.shape  # weight: (IN_C, OUT_C, KD, KH, KW)
    
    STRIDE_D, STRIDE_H, STRIDE_W = 2, 2, 2
    PAD_D, PAD_H, PAD_W = 1, 1, 1
    OUT_PAD_D, OUT_PAD_H, OUT_PAD_W = 1, 1, 1
    
    OUT_D = (IN_D - 1) * STRIDE_D - 2 * PAD_D + KD + OUT_PAD_D
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH + OUT_PAD_H
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW + OUT_PAD_W
    
    # Allocate conv output
    conv_out = torch.empty(N, OUT_C, OUT_D, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M = N * OUT_D * OUT_H * OUT_W
    K = IN_C * KD * KH * KW
    
    # Strides for x: (N, IN_C, IN_D, IN_H, IN_W)
    s_xn, s_xc, s_xd, s_xh, s_xw = x.stride()
    # Weight: (IN_C, OUT_C, KD, KH, KW)
    s_wic, s_woc, s_wkd, s_wkh, s_wkw = p_conv_transpose_weight.stride()
    # Output: (N, OUT_C, OUT_D, OUT_H, OUT_W)
    s_yn, s_yc, s_yd, s_yh, s_yw = conv_out.stride()
    
    grid_conv = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv_transpose3d_kernel[grid_conv](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        N, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M, OUT_C, K,
        s_xn, s_xc, s_xd, s_xh, s_xw,
        s_wic, s_woc, s_wkd, s_wkh, s_wkw,
        s_yn, s_yc, s_yd, s_yh, s_yw,
    )
    
    # MaxPool3d parameters
    POOL_KD, POOL_KH, POOL_KW = 2, 2, 2
    POOL_SD, POOL_SH, POOL_SW = 2, 2, 2
    
    POOL_OUT_D = (OUT_D - POOL_KD) // POOL_SD + 1
    POOL_OUT_H = (OUT_H - POOL_KH) // POOL_SH + 1
    POOL_OUT_W = (OUT_W - POOL_KW) // POOL_SW + 1
    
    SPATIAL = POOL_OUT_D * POOL_OUT_H * POOL_OUT_W
    
    out_vals = torch.empty(N, POOL_OUT_D, POOL_OUT_H, POOL_OUT_W, device=x.device, dtype=x.dtype)
    out_idxs = torch.empty(N, POOL_OUT_D, POOL_OUT_H, POOL_OUT_W, device=x.device, dtype=torch.int64)
    
    total_programs = N * SPATIAL
    
    grid_fused = lambda meta: (total_programs,)
    
    pool_softmax_swish_max_kernel[grid_fused](
        conv_out,
        p_subtract,
        out_vals,
        out_idxs,
        N, OUT_C,
        OUT_D, OUT_H, OUT_W,
        POOL_OUT_D, POOL_OUT_H, POOL_OUT_W,
        POOL_KD, POOL_KH, POOL_KW,
        POOL_SD, POOL_SH, POOL_SW,
        SPATIAL,
        s_yn, s_yc, s_yd, s_yh, s_yw,
    )
    
    return out_vals, out_idxs


def get_inputs():
    torch.manual_seed(42)
    p_conv_transpose_weight = torch.randn(3, 16, 3, 3, 3)
    p_conv_transpose_bias = torch.randn(16)
    p_subtract = torch.randn(16)
    x = torch.rand(128, 3, 16, 32, 32)
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_subtract, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs_cpu = get_inputs()
    inputs = [t.cuda() for t in inputs_cpu]
    
    out_vals, out_idxs = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        p_conv_transpose_weight, p_conv_transpose_bias, p_subtract, x = inputs
        
        # Reference computation
        conv_t = nn.ConvTranspose3d(3, 16, 3, stride=2, padding=1, output_padding=1).cuda()
        conv_t.weight.data = p_conv_transpose_weight
        conv_t.bias.data = p_conv_transpose_bias
        
        with torch.no_grad():
            ref = conv_t(x)
            ref = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(ref)
            ref = torch.softmax(ref, dim=1)
            ref = ref - p_subtract.view(1, -1, 1, 1, 1)
            ref = torch.sigmoid(ref) * ref
            ref_vals, ref_idxs = torch.max(ref, dim=1)
        
        print(f"Max diff values: {(out_vals - ref_vals).abs().max().item():.4e}")
        print(f"Max diff indices: {(out_idxs - ref_idxs).abs().max().item():.4e}")
        
        assert torch.allclose(out_vals, ref_vals, rtol=1e-3, atol=1e-5), \
            f"Values don't match! max_diff={(out_vals - ref_vals).abs().max().item():.4e}"
        print("Kernel executed successfully.")