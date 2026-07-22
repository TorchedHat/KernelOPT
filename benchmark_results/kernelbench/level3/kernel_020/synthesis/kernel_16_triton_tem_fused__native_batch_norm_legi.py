import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + optional ReLU6 kernel (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    apply_relu6: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    
    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Generic Conv2d Triton kernel (implicit GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    # Input dims
    BATCH, IN_C, IN_H, IN_W,
    # Output dims
    OUT_C, OUT_H, OUT_W,
    # Kernel
    KH, KW,
    # Stride
    SH, SW,
    # Padding
    PH, PW,
    # Groups
    GROUPS,
    # Total sizes
    M, N, K,
    # Strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # Strides for W (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_wo, s_wkh, s_wkw, s_wc,
    # Strides for Out (NHWC)
    s_on, s_oh, s_ow, s_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # Group handling
    group_size_out = OUT_C // GROUPS
    group_size_in = IN_C // GROUPS
    group_idx = offs_n // group_size_out
    local_out_c = offs_n % group_size_out
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = group_size_in * KH * KW
    K_per_group = group_size_in * KH * KW
    
    for k_start in range(0, K_per_group, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (in_c_local, kh, kw)
        in_c_local = k_offs % group_size_in
        kh_idx = (k_offs // group_size_in) % KH
        kw_idx = k_offs // (group_size_in * KH)
        
        # Input coordinates
        in_h_idx = out_h_idx[:, None] * SH + kh_idx[None, :] - PH
        in_w_idx = out_w_idx[:, None] * SW + kw_idx[None, :] - PW
        in_c_abs = group_idx[None, :] * group_size_in + in_c_local[None, :]  # shape [1, BLOCK_K] broadcast
        
        # For grouped conv, we need per-output-channel group
        # Actually group_idx depends on offs_n (output channel), not k
        # Let's handle this differently - for each output channel group
        # We need to load input from the corresponding input channel group
        
        # Simplified: assume group_idx is same for all offs_n in this block
        # This works when BLOCK_N <= group_size_out
        # For depthwise (groups=IN_C), group_size_out=1, group_size_in=1
        
        # in_c_abs for loading X: depends on which group each output channel belongs to
        # group_idx shape: [BLOCK_N], in_c_local shape: [BLOCK_K]
        # We need in_c_abs[m, k] = group_idx[n] * group_size_in + in_c_local[k]
        # But m and n are different dims...
        
        # For the X load, we need [BLOCK_M, BLOCK_K] where each row uses the same in_c
        # For grouped conv, different output channels may belong to different groups
        # This is complex - let's use a simpler approach for non-grouped first
        
        # X load: [BLOCK_M, BLOCK_K]
        # For group g, input channels are [g*group_size_in, (g+1)*group_size_in)
        # Since all offs_n in a block may span multiple groups, this is tricky
        # For simplicity, assume single group or depthwise handled separately
        
        in_c_for_x = in_c_local  # [BLOCK_K] - local channel within group
        # We'll use group 0 for now and handle groups=1 case
        # For depthwise, we need special handling
        
        x_h_valid = (in_h_idx >= 0) & (in_h_idx < IN_H)
        x_w_valid = (in_w_idx >= 0) & (in_w_idx < IN_W)
        m_valid = (offs_m[:, None] < M) & (batch_idx[:, None] < BATCH)
        k_valid = k_offs[None, :] < K_per_group
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * s_xn +
                  in_c_for_x[None, :] * s_xc)
        
        x_mask = m_valid & x_h_valid & x_w_valid & k_valid
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W load: [BLOCK_K, BLOCK_N]
        # W shape: (OUT_C, group_size_in, KH, KW) in NCHW
        # Stored as (out_c, kh, kw, in_c) for our access pattern
        n_valid = offs_n[None, :] < N
        
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wo +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw +
                  in_c_local[:, None] * s_wc)
        
        w_mask = (k_offs[:, None] < K_per_group) & n_valid
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * s_on +
                out_h_idx[:, None] * s_oh +
                out_w_idx[:, None] * s_ow +
                offs_n[None, :] * s_oc)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Linear layer kernel (GEMM: [M, K] x [K, N] -> [M, N])
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * K + k_offs[None, :]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W is [N, K], so W[n, k]
        w_ptrs = W_ptr + offs_n[:, None] * K + k_offs[None, :]
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile.T, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_bn_relu6(x, mean, var, weight, bias, apply_relu6=True, eps=1e-5):
    """Apply batch norm + optional relu6 using Triton."""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    # Reshape to NHWC for channel-indexed access
    # x is in NCHW, we need channel index = (flat_idx // spatial) % C
    # For NCHW: flat_idx = n*C*H*W + c*H*W + h*W + w
    # channel = (flat_idx // (H*W)) % C -- this works for NCHW!
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial,
        apply_relu6, eps,
    )
    return out


def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    """
    """
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = w.shape
    SH, SW = stride
    PH, PW = padding
    
    OUT_H = (H + 2*PH - KH) // SH + 1
    OUT_W = (W + 2*PW - KW) // SW + 1
    
    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # N, H, W, C
    
    # Convert w to (out_c, kh, kw, in_c_per_group) for our kernel
    w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # out_c, kh, kw, in_c_per_group
    
    M = N * OUT_H * OUT_W
    N_out = C_out
    K = C_in_per_group * KH * KW
    
    # Output in NHWC
    out_nhwc = torch.empty(N, OUT_H, OUT_W, C_out, device=x.device, dtype=x.dtype)
    
    # Strides for x_nhwc: (N, H, W, C) -> strides (H*W*C, W*C, C, 1)
    s_xn = H * W * C_in
    s_xh = W * C_in
    s_xw = C_in
    s_xc = 1
    
    # Strides for w_nhwc: (out_c, kh, kw, in_c) -> strides (kh*kw*in_c, kw*in_c, in_c, 1)
    s_wo = KH * KW * C_in_per_group
    s_wkh = KW * C_in_per_group
    s_wkw = C_in_per_group
    s_wc = 1
    
    # Strides for out_nhwc: (N, H, W, C) -> strides (OUT_H*OUT_W*C_out, OUT_W*C_out, C_out, 1)
    s_on = OUT_H * OUT_W * C_out
    s_oh = OUT_W * C_out
    s_ow = C_out
    s_oc = 1
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x_nhwc, w_nhwc, out_nhwc,
        N, C_in, H, W,
        C_out, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        groups,
        M, N_out, K,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wc,
        s_on, s_oh, s_ow, s_oc,
    )
    
    # Convert back to NCHW
    return out_nhwc.permute(0, 3, 1, 2).contiguous()


def triton_linear(x, w, b):
    """
    w: [N, K]  (weight matrix, N=out_features, K=in_features)
    """
    M, K = x.shape
    N = w.shape[0]
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
    )
    return out


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_features_3_0_weight,
    p_features_3_1_weight,
    p_features_3_1_bias,
    p_features_3_3_weight,
    p_features_3_4_weight,
    p_features_3_4_bias,
    p_features_4_0_weight,
    p_features_4_1_weight,
    p_features_4_1_bias,
    p_features_4_3_weight,
    p_features_4_4_weight,
    p_features_4_4_bias,
    p_features_4_6_weight,
    p_features_4_7_weight,
    p_features_4_7_bias,
    p_features_5_0_weight,
    p_features_5_1_weight,
    p_features_5_1_bias,
    p_features_5_3_weight,
    p_features_5_4_weight,
    p_features_5_4_bias,
    p_features_5_6_weight,
    p_features_5_7_weight,
    p_features_5_7_bias,
    p_features_6_0_weight,
    p_features_6_1_weight,
    p_features_6_1_bias,
    p_features_6_3_weight,
    p_features_6_4_weight,
    p_features_6_4_bias,
    p_features_6_6_weight,
    p_features_6_7_weight,
    p_features_6_7_bias,
    p_features_7_0_weight,
    p_features_7_1_weight,
    p_features_7_1_bias,
    p_features_7_3_weight,
    p_features_7_4_weight,
    p_features_7_4_bias,
    p_features_7_6_weight,
    p_features_7_7_weight,
    p_features_7_7_bias,
    p_features_8_0_weight,
    p_features_8_1_weight,
    p_features_8_1_bias,
    p_features_8_3_weight,
    p_features_8_4_weight,
    p_features_8_4_bias,
    p_features_8_6_weight,
    p_features_8_7_weight,
    p_features_8_7_bias,
    p_features_9_0_weight,
    p_features_9_1_weight,
    p_features_9_1_bias,
    p_features_9_3_weight,
    p_features_9_4_weight,
    p_features_9_4_bias,
    p_features_9_6_weight,
    p_features_9_7_weight,
    p_features_9_7_bias,
    p_features_10_0_weight,
    p_features_10_1_weight,
    p_features_10_1_bias,
    p_features_10_3_weight,
    p_features_10_4_weight,
    p_features_10_4_bias,
    p_features_10_6_weight,
    p_features_10_7_weight,
    p_features_10_7_bias,
    p_features_11_0_weight,
    p_features_11_1_weight,
    p_features_11_1_bias,
    p_features_11_3_weight,
    p_features_11_4_weight,
    p_features_11_4_bias,
    p_features_11_6_weight,
    p_features_11_7_weight,
    p_features_11_7_bias,
    p_features_12_0_weight,
    p_features_12_1_weight,
    p_features_12_1_bias,
    p_features_12_3_weight,
    p_features_12_4_weight,
    p_features_12_4_bias,
    p_features_12_6_weight,
    p_features_12_7_weight,
    p_features_12_7_bias,
    p_features_13_0_weight,
    p_features_13_1_weight,
    p_features_13_1_bias,
    p_features_13_3_weight,
    p_features_13_4_weight,
    p_features_13_4_bias,
    p_features_13_6_weight,
    p_features_13_7_weight,
    p_features_13_7_bias,
    p_features_14_0_weight,
    p_features_14_1_weight,
    p_features_14_1_bias,
    p_features_14_3_weight,
    p_features_14_4_weight,
    p_features_14_4_bias,
    p_features_14_6_weight,
    p_features_14_7_weight,
    p_features_14_7_bias,
    p_features_15_0_weight,
    p_features_15_1_weight,
    p_features_15_1_bias,
    p_features_15_3_weight,
    p_features_15_4_weight,
    p_features_15_4_bias,
    p_features_15_6_weight,
    p_features_15_7_weight,
    p_features_15_7_bias,
    p_features_16_0_weight,
    p_features_16_1_weight,
    p_features_16_1_bias,
    p_features_16_3_weight,
    p_features_16_4_weight,
    p_features_16_4_bias,
    p_features_16_6_weight,
    p_features_16_7_weight,
    p_features_16_7_bias,
    p_features_17_0_weight,
    p_features_17_1_weight,
    p_features_17_1_bias,
    p_features_17_3_weight,
    p_features_17_4_weight,
    p_features_17_4_bias,
    p_features_17_6_weight,
    p_features_17_7_weight,
    p_features_17_7_bias,
    p_features_18_0_weight,
    p_features_18_1_weight,
    p_features_18_1_bias,
    p_features_18_3_weight,
    p_features_18_4_weight,
    p_features_18_4_bias,
    p_features_18_6_weight,
    p_features_18_7_weight,
    p_features_18_7_bias,
    p_features_19_0_weight,
    p_features_19_1_weight,
    p_features_19_1_bias,
    p_features_19_3_weight,
    p_features_19_4_weight,
    p_features_19_4_bias,
    p_features_19_6_weight,
    p_features_19_7_weight,
    p_features_19_7_bias,
    p_features_20_weight,
    p_features_21_weight,
    p_features_21_bias,
    p_classifier_1_weight,
    p_classifier_1_bias,
    b_features_1_running_mean,
    b_features_1_running_var,
    b_features_3_1_running_mean,
    b_features_3_1_running_var,
    b_features_3_4_running_mean,
    b_features_3_4_running_var,
    b_features_4_1_running_mean,
    b_features_4_1_running_var,
    b_features_4_4_running_mean,
    b_features_4_4_running_var,
    b_features_4_7_running_mean,
    b_features_4_7_running_var,
    b_features_5_1_running_mean,
    b_features_5_1_running_var,
    b_features_5_4_running_mean,
    b_features_5_4_running_var,
    b_features_5_7_running_mean,
    b_features_5_7_running_var,
    b_features_6_1_running_mean,
    b_features_6_1_running_var,
    b_features_6_4_running_mean,
    b_features_6_4_running_var,
    b_features_6_7_running_mean,
    b_features_6_7_running_var,
    b_features_7_1_running_mean,
    b_features_7_1_running_var,
    b_features_7_4_running_mean,
    b_features_7_4_running_var,
    b_features_7_7_running_mean,
    b_features_7_7_running_var,
    b_features_8_1_running_mean,
    b_features_8_1_running_var,
    b_features_8_4_running_mean,
    b_features_8_4_running_var,
    b_features_8_7_running_mean,
    b_features_8_7_running_var,
    b_features_9_1_running_mean,
    b_features_9_1_running_var,
    b_features_9_4_running_mean,
    b_features_9_4_running_var,
    b_features_9_7_running_mean,
    b_features_9_7_running_var,
    b_features_10_1_running_mean,
    b_features_10_1_running_var,
    b_features_10_4_running_mean,
    b_features_10_4_running_var,
    b_features_10_7_running_mean,
    b_features_10_7_running_var,
    b_features_11_1_running_mean,
    b_features_11_1_running_var,
    b_features_11_4_running_mean,
    b_features_11_4_running_var,
    b_features_11_7_running_mean,
    b_features_11_7_running_var,
    b_features_12_1_running_mean,