import torch
import triton
import triton.language as tl

# ConvTranspose3d kernel using scatter approach
# For each output position, we accumulate contributions from input positions
# ConvTranspose3d: output[n, co, od, oh, ow] += input[n, ci, id, ih, iw] * weight[ci, co, kd, kh, kw]
# where od = id*stride + kd - pad, etc.
# Equivalently: for each (n, ci, id, ih, iw), scatter to output positions

# We implement it as: output = input_unfolded @ weight_reshaped
# Using the GEMM approach: treat it as a matrix multiply
# M = N * OD * OH * OW, N_out = CO, K = CI * KD * KH * KW
# But ConvTranspose is the transpose, so:
# M = N * ID * IH * IW (input spatial), K = CI, N_out = CO * KD * KH * KW
# Then scatter-add to output

# Better approach: implement as standard GEMM where
# For each output pixel (n, od, oh, ow, co), sum over (ci, kd, kh, kw):
#   out[n,co,od,oh,ow] += x[n, ci, (od-kd)//stride, (oh-kh)//stride, (ow-kw)//stride] * w[ci, co, kd, kh, kw]
# where (od-kd) must be divisible by stride and in bounds

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N_OUT', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    N, CI, ID, IH, IW,
    CO, OD, OH, OW,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M, N_OUT, K,
    # X strides: (N, CI, ID, IH, IW)
    x_sn, x_sc, x_sd, x_sh, x_sw,
    # W strides: (CI, CO, KD, KH, KW)
    w_sci, w_sco, w_skd, w_skh, w_skw,
    # OUT strides: (N, CO, OD, OH, OW)
    o_sn, o_sc, o_sd, o_sh, o_sw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    M = N * OD * OH * OW (output spatial positions)
    N_OUT = CO (output channels)
    K = CI * KD * KH * KW (reduction: input channels * kernel volume)
    
    out[m, n_out] = sum over k=(ci, kd, kh, kw):
        x[n, ci, id, ih, iw] * w[ci, co, kd, kh, kw]
    where id = (od - kd) / STRIDE_D (must be integer and in bounds)
          ih = (oh - kh) / STRIDE_H
          iw = (ow - kw) / STRIDE_W
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (n, od, oh, ow)
    ow_idx = offs_m % OW
    oh_idx = (offs_m // OW) % OH
    od_idx = (offs_m // (OW * OH)) % OD
    n_idx = offs_m // (OW * OH * OD)
    
    # Decode n -> co
    co_idx = offs_n  # shape [BLOCK_N]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ci, kd, kh, kw)
        kw_k = offs_k % KW
        kh_k = (offs_k // KW) % KH
        kd_k = (offs_k // (KW * KH)) % KD
        ci_k = offs_k // (KW * KH * KD)
        
        # For ConvTranspose: od = id * STRIDE_D + kd
        # So id = (od - kd) / STRIDE_D, must be integer and in [0, ID)
        # od_idx: [BLOCK_M], kd_k: [BLOCK_K]
        od_m = od_idx[:, None]  # [BLOCK_M, 1]
        oh_m = oh_idx[:, None]
        ow_m = ow_idx[:, None]
        n_m = n_idx[:, None]
        
        kd_k2 = kd_k[None, :]  # [1, BLOCK_K]
        kh_k2 = kh_k[None, :]
        kw_k2 = kw_k[None, :]
        ci_k2 = ci_k[None, :]
        
        # Compute input indices
        id_num = od_m - kd_k2  # [BLOCK_M, BLOCK_K]
        ih_num = oh_m - kh_k2
        iw_num = ow_m - kw_k2
        
        # Check divisibility by stride and bounds
        id_valid_div = (id_num % STRIDE_D) == 0
        ih_valid_div = (ih_num % STRIDE_H) == 0
        iw_valid_div = (iw_num % STRIDE_W) == 0
        
        id_in = id_num // STRIDE_D
        ih_in = ih_num // STRIDE_H
        iw_in = iw_num // STRIDE_W
        
        valid_m = (offs_m[:, None] < M)
        valid_k = (offs_k[None, :] < K)
        valid_id = (id_in >= 0) & (id_in < ID)
        valid_ih = (ih_in >= 0) & (ih_in < IH)
        valid_iw = (iw_in >= 0) & (iw_in < IW)
        valid_ci = (ci_k2 < CI)
        
        x_mask = valid_m & valid_k & id_valid_div & ih_valid_div & iw_valid_div & valid_id & valid_ih & valid_iw & valid_ci
        
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight: w[ci, co, kd, kh, kw]
        # offs_k: [BLOCK_K], offs_n: [BLOCK_N]
        w_ptrs = (W_ptr
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_OUT)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_OUT, other=0.0)
    acc = acc + bias[None, :]
    
    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    out_ptrs = (OUT_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['spatial_size'],
)
@triton.jit
def scale_bn_avgpool_kernel(
    X_ptr, OUT_ptr,
    BN_weight_ptr, BN_bias_ptr,
    BN_mean_ptr, BN_var_ptr,
    scale_factor,
    N, C, spatial_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    n_idx = pid // C
    c_idx = pid % C
    
    # Load BN params
    bn_mean = tl.load(BN_mean_ptr + c_idx).to(tl.float32)
    bn_var = tl.load(BN_var_ptr + c_idx).to(tl.float32)
    bn_w = tl.load(BN_weight_ptr + c_idx).to(tl.float32)
    bn_b = tl.load(BN_bias_ptr + c_idx).to(tl.float32)
    
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    
    base_ptr = X_ptr + n_idx * C * spatial_size + c_idx * spatial_size
    
    acc = tl.zeros((), dtype=tl.float32)
    
    for off in range(0, spatial_size, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial_size
        x_val = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Scale
        x_val = x_val * scale_factor
        # BN inference: (x - mean) / sqrt(var + eps) * weight + bias
        x_norm = (x_val - bn_mean) * inv_std * bn_w + bn_b
        # Accumulate for avg pool
        acc += tl.sum(tl.where(mask, x_norm, 0.0), axis=0)
    
    avg = acc / spatial_size
    
    # Output: (N, C, 1, 1, 1) -> linear index
    out_idx = n_idx * C + c_idx
    tl.store(OUT_ptr + out_idx, avg)


def kernel_function(
    p_conv_transpose_weight,
    p_conv_transpose_bias,
    p_batch_norm_weight,
    p_batch_norm_bias,
    b_batch_norm_running_mean,
    b_batch_norm_running_var,
    x,
):
    # Input: x shape (N, CI, ID, IH, IW)
    N, CI, ID, IH, IW = x.shape
    CO, _, KD, KH, KW = p_conv_transpose_weight.shape  # weight: (CI, CO, KD, KH, KW)
    # Wait, ConvTranspose3d weight shape is (in_channels, out_channels, kD, kH, kW)
    # p_conv_transpose_weight: shape=[64, 128, 5, 5, 5] -> (CI=64, CO=128, KD=5, KH=5, KW=5)
    CI_w, CO, KD, KH, KW = p_conv_transpose_weight.shape
    
    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0
    
    # Output size for ConvTranspose3d (no padding, stride=1):
    # OD = (ID - 1) * STRIDE_D - 2*PAD_D + KD
    OD = (ID - 1) * STRIDE_D - 2 * PAD_D + KD
    OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH
    OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW
    
    # Allocate output
    conv_out = torch.empty((N, CO, OD, OH, OW), device=x.device, dtype=x.dtype)
    
    M = N * OD * OH * OW
    N_OUT = CO
    K = CI * KD * KH * KW
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_OUT, meta['BLOCK_N']),
    )
    
    conv_transpose3d_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        N, CI, ID, IH, IW,
        CO, OD, OH, OW,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M, N_OUT, K,
        # X strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        # W strides: (CI, CO, KD, KH, KW)
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        p_conv_transpose_weight.stride(4),
        # OUT strides
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
    )
    
    # Fused scale + BN + global avg pool
    spatial_size = OD * OH * OW
    pool_out = torch.empty((N, CO, 1, 1, 1), device=x.device, dtype=x.dtype)
    pool_out_flat = pool_out.view(N * CO)
    
    scale_bn_avgpool_kernel[(N * CO,)](
        conv_out, pool_out_flat,
        p_batch_norm_weight, p_batch_norm_bias,
        b_batch_norm_running_mean, b_batch_norm_running_var,
        2.0,  # scale_factor
        N, CO, spatial_size,
    )
    
    return pool_out


def get_inputs():
    p_conv_transpose_weight = torch.randn(64, 128, 5, 5, 5)
    p_conv_transpose_bias = torch.randn(128)
    p_batch_norm_weight = torch.randn(128)
    p_batch_norm_bias = torch.randn(128)
    b_batch_norm_running_mean = torch.randn(128)
    b_batch_norm_running_var = torch.rand(128) + 0.1  # positive
    x = torch.rand(16, 64, 16, 32, 32)
    return [
        p_conv_transpose_weight,
        p_conv_transpose_bias,
        p_batch_norm_weight,
        p_batch_norm_bias,
        b_batch_norm_running_mean,
        b_batch_norm_running_var,
        x,
    ]


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
        (p_conv_transpose_weight, p_conv_transpose_bias,
         p_batch_norm_weight, p_batch_norm_bias,
         b_batch_norm_running_mean, b_batch_norm_running_var, x) = inputs_cuda
        
        # Reference computation
        import torch.nn as nn
        conv_t = nn.ConvTranspose3d(64, 128, 5, bias=True).cuda()
        conv_t.weight.data = p_conv_transpose_weight
        conv_t.bias.data = p_conv_transpose_bias
        
        bn = nn.BatchNorm3d(128, eps=1e-5).cuda()
        bn.weight.data = p_batch_norm_weight
        bn.bias.data = p_batch_norm_bias
        bn.running_mean.data = b_batch_norm_running_mean
        bn.running_var.data = b_batch_norm_running_var
        bn.eval()
        
        with torch.no_grad():
            ref = conv_t(x)
            ref = ref * 2.0
            ref = bn(ref)
            ref = ref.mean(dim=(2, 3, 4), keepdim=True)
        
        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")