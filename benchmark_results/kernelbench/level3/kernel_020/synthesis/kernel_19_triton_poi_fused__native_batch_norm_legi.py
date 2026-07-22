import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU6 (hardtanh(0,6)) pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_channels', 'numel'],
)
@triton.jit
def bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Channel index: tensor is (N, C, H, W) in NCHW layout
    # offsets index into flattened tensor, channel = (offset // (H*W)) % C
    # But we pass spatial_size = H*W so channel = (offset // spatial_size) % n_channels
    # Actually numel = N*C*H*W, channel index = (offset / (numel/(N*C))) % C
    # Simpler: channel = (offset // spatial_size) % n_channels
    # We'll compute it from offsets directly
    # offset = n * C * H * W + c * H * W + h * W + w
    # channel = (offset // (H*W)) % C -- but we don't have H*W here
    # Instead use: channel = (offset % (n_channels * spatial_size)) // spatial_size
    # We'll pass spatial_size separately... let's use numel // n_channels approach
    # Actually simplest: channel index in NCHW = (offset // spatial_size) % n_channels
    # We need spatial_size. Let's compute: spatial_size = numel / (N * n_channels)
    # But N is not passed. Let's just pass spatial_size.
    # For now, restructure to pass spatial_size.
    pass

# Better approach: pass spatial_size
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_channels', 'spatial_size'],
)
@triton.jit
def fused_bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, spatial_size,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    numel = n_channels * spatial_size  # per-batch, but we handle all batches
    # Actually total numel = N * C * H * W
    # We'll index globally
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n_channels * spatial_size  # This is wrong for batched
    # Let total_numel be passed... let me restructure

    pass


# Let me write the actual clean implementation

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_channels', 'spatial_size'],
)
@triton.jit
def fused_bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, spatial_size, total_numel,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Input tensor: (N, C, H, W) contiguous, total_numel = N*C*H*W
    channel index = (flat_idx // spatial_size) % n_channels
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute channel index
    c_idx = (offsets // spatial_size) % n_channels

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    eps_val = 1e-5
    x_norm = (x - mean) / tl.sqrt(var + eps_val)
    out = x_norm * w + b

    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn_act(x, mean, var, weight, bias, apply_relu6=True):
    """Apply fused BN + optional ReLU6 using Triton kernel."""
    N, C, H, W = x.shape
    spatial_size = H * W
    total_numel = N * C * H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(total_numel, meta['BLOCK_SIZE']),)
    fused_bn_act_kernel[grid](
        x, out,
        mean, var, weight, bias,
        C, spatial_size, total_numel,
        apply_relu6,
    )
    return out


# Implicit GEMM Conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W', 'GROUPS'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (groups, out_c_per_group, kh, kw, in_c_per_group)
    s_wg, s_wo, s_wkh, s_wkw, s_wi,
    # strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = (IN_C // GROUPS) * KERNEL_H * KERNEL_W
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    TOTAL_K = GROUP_IN_C * KERNEL_H * KERNEL_W
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # Which group does each output channel belong to?
    group_idx = offs_n // GROUP_OUT_C  # shape [BLOCK_N]
    local_out_c = offs_n % GROUP_OUT_C  # local channel within group
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        kw = (offs_k // GROUP_IN_C) % KERNEL_W
        kh = offs_k // (GROUP_IN_C * KERNEL_W)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        # Global input channel (group * GROUP_IN_C + in_c_local)
        # For each output channel n, group = n // GROUP_OUT_C
        # But we need per-M group... actually group is determined by output channel
        # For depthwise: GROUPS=IN_C=OUT_C, GROUP_IN_C=1, GROUP_OUT_C=1
        # group for output n: group_idx[n]
        # input channel: group_idx[n] * GROUP_IN_C + in_c_local[k]
        # This creates a dependency between M-dim and N-dim for group selection
        # For simplicity, handle groups=1 and groups=IN_C (depthwise) cases
        
        # For standard conv (groups=1): in_c = in_c_local
        # For depthwise (groups=IN_C): group = out_c, in_c = 0 (GROUP_IN_C=1)
        # General: in_c = group * GROUP_IN_C + in_c_local
        # group depends on output channel (N dim), not M dim
        # So: in_c[m, k, n] = group_idx[n] * GROUP_IN_C + in_c_local[k]
        
        in_c_global = group_idx[None, :] * GROUP_IN_C + in_c_local[:, None]  # [BLOCK_K, BLOCK_N]
        
        # Load X: shape [BLOCK_M, BLOCK_K]
        # x[batch, ih, iw, in_c_global] but in_c_global depends on N...
        # This is problematic for a standard GEMM structure
        # For groups=1: in_c_global is just in_c_local (independent of N)
        # For depthwise: GROUP_IN_C=1, so in_c_global = group_idx (= out_c)
        
        # Handle groups=1 case (standard conv):
        valid_k = offs_k < TOTAL_K
        valid_m = offs_m < M_TOTAL
        valid_n = offs_n < OUT_C
        
        ih_valid = (ih >= 0) & (ih < IN_H)
        iw_valid = (iw >= 0) & (iw < IN_W)
        
        # For groups=1, load X normally
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * s_xn + 
                  in_c_local[None, :] * s_xc)
        
        x_mask = (valid_m[:, None] & ih_valid & iw_valid & valid_k[None, :])
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W: shape [BLOCK_K, BLOCK_N]
        # w[group, local_out_c, kh, kw, in_c_local]
        w_ptrs = (W_ptr +
                  group_idx[None, :] * s_wg +
                  local_out_c[None, :] * s_wo +
                  kh[:, None] * s_wkh +
                  kw[:, None] * s_wkw +
                  in_c_local[:, None] * s_wi)
        
        w_mask = (valid_k[:, None] & valid_n[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C
    out_mask = valid_m[:, None] & valid_n[None, :]
    
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              offs_n[None, :] * s_yc)
    
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    """
    returns: (N, OUT_C, OH, OW)
    """
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    
    stride_h, stride_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1
    
    GROUP_IN_C = IN_C // groups
    GROUP_OUT_C = OUT_C // groups
    
    # Convert to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    
    # Reshape weight: (groups, GROUP_OUT_C, KH, KW, GROUP_IN_C)
    w_grouped = w.reshape(groups, GROUP_OUT_C, GROUP_IN_C, KH, KW).permute(0, 1, 3, 4, 2).contiguous()
    # w_grouped: (groups, GROUP_OUT_C, KH, KW, GROUP_IN_C)
    
    # Output in NHWC
    y_nhwc = torch.empty(N, OUT_H, OUT_W, OUT_C, device=x.device, dtype=x.dtype)
    
    M_TOTAL = N * OUT_H * OUT_W
    
    # Strides for x_nhwc (N, H, W, C)
    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    # Strides for w_grouped (groups, GROUP_OUT_C, KH, KW, GROUP_IN_C)
    s_wg, s_wo, s_wkh, s_wkw, s_wi = w_grouped.stride()
    # Strides for y_nhwc (N, H, W, C)
    s_yn, s_yh, s_yw, s_yc = y_nhwc.stride()
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x_nhwc, w_grouped, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        groups,
        s_xn, s_xh, s_xw, s_xc,
        s_wg, s_wo, s_wkh, s_wkw, s_wi,
        s_yn, s_yh, s_yw, s_yc,
    )
    
    # Convert back to NCHW
    return y_nhwc.permute(0, 3, 1, 2).contiguous()


# Depthwise conv kernel (groups = IN_C = OUT_C)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 9}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 9}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 9}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 9}, num_warps=2, num_stages=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    s_xn, s_xc, s_xh, s_xw,
    s_wc, s_wkh, s_wkw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Depthwise conv2d: groups = IN_C = OUT_C
    pid_c = channel index
    pid_m = spatial tile (batch * OUT_H * OUT_W)
    """
    pid_c = tl.program_id(1)
    pid_m = tl.program_id(0)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_m = offs_m < M_TOTAL
    
    # Decode spatial
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # Load kernel weights for this channel: KH * KW values
    TOTAL_K = KERNEL_H * KERNEL_W
    offs_k = tl.arange(0, BLOCK_K)
    kw = offs_k % KERNEL_W
    kh = offs_k // KERNEL_W
    
    valid_k = offs_k < TOTAL_K
    valid_c = pid_c < IN_C
    
    w_ptrs = W_ptr + pid_c * s_wc + kh * s_wkh + kw * s_wkw
    w_vals = tl.load(w_ptrs, mask=valid_k & valid_c, other=0.0)  # [BLOCK_K]
    
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for k in range(TOTAL_K):
        kh_val = k // KERNEL_W
        kw_val = k % KERNEL_W
        
        ih = out_h_idx * STRIDE_H + kh_val - PAD_H
        iw = out_w_idx * STRIDE_W + kw_val - PAD_W
        
        ih_valid = (ih >= 0) & (ih < IN_H)
        iw_valid = (iw >= 0) & (iw < IN_W)
        
        x_ptrs = X_ptr + batch_idx * s_xn + pid_c * s_xc + ih * s_xh + iw * s_xw
        x_vals = tl.load(x_ptrs, mask=valid_m & ih_valid & iw_valid & valid_c, other=0.0)
        
        w_val = tl.load(W_ptr + pid_c * s_wc + kh_val * s_wkh + kw_val * s_wkw,
                        mask=valid_c, other=0.0)
        
        acc += x_vals * w_val
    
    # Store
    y_ptrs = Y_ptr + batch_idx * s_yn + pid_c * s_yc + out_h_idx * s_yh + out_w_idx * s_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_m & valid_c)


def triton_depthwise_conv2d(x, w, stride=(1,1), padding=(1,1)):
    """
    Depthwise conv2d: groups = IN_C = OUT_C
    """
    N, C, H, W = x.shape
    _, _, KH, KW = w.shape
    
    stride_h, stride_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    OUT_H = (H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (W + 2 * pad_w - KW) // stride_w + 1
    
    # w: (C, 1, KH, KW) -> (C, KH, KW)
    w_dw = w.squeeze(1).contiguous()
    
    y = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wc, s_wkh, s_wkw = w_dw.stride()
    s_yn, s_yc, s_yh, s_yw = y.stride()
    
    M_TOTAL = N * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        C,
    )
    
    depthwise_conv2d_kernel[grid](
        x, w_dw, y,
        N, C, H, W,
        OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        s_xn, s_xc, s_xh, s_xw,
        s_wc, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
    )
    
    return y


# Pointwise conv (1x1) using matrix multiply via tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def pointwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    1x1 conv as matrix multiply: Y = X @ W^T
    X: (M, K) where M = N_batch * H * W, K = IN_C
    Y: (M, N) where N = OUT_C
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W is (OUT_C, IN_C), we want W[n, k] -> load W[offs_n, offs_k]
        w_ptrs = W_ptr + offs_n[None, :