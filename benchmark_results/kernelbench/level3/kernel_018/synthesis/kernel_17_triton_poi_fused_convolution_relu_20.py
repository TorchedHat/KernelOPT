import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused bias add + ReLU (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_kernel(
    n_elements,  # total elements = N*C*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index: offset // hw % n_channels
    # For NCHW layout: index = n*C*H*W + c*H*W + h*W + w
    # channel = (offset // hw) % n_channels
    chan = (offsets // hw) % n_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    result = x + b
    result = tl.where(result > 0.0, result, 0.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (kernel=3, stride=2, ceil_mode=True)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def max_pool2d_k3s2_kernel(
    N, C,
    H_in, W_in,
    H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode output index (NCHW)
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c_idx = (offsets // (W_out * H_out)) % C
    n_idx = offsets // (W_out * H_out * C)
    
    # Input base for this (n, c)
    in_base = n_idx * C * H_in * W_in + c_idx * H_in * W_in
    
    # Pool window: h_in_start = h_out*2, w_in_start = w_out*2
    h_start = h_out * 2
    w_start = w_out * 2
    
    # Load 9 values (3x3 window) with boundary checks
    neg_inf = float('-inf')
    
    val = tl.full([BLOCK_SIZE], neg_inf, dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = h_start + kh
            iw = w_start + kw
            valid = mask & (ih < H_in) & (iw < W_in)
            in_idx = in_base + ih * W_in + iw
            v = tl.load(x_ptr + in_idx, mask=valid, other=neg_inf)
            val = tl.where(valid & (v > val), v, val)
    
    tl.store(out_ptr + offsets, val, mask=mask)


# ============================================================
# Triton kernel: fused conv2d (1x1) + bias + ReLU
# Uses tl.dot for the matrix multiply
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_relu_kernel(
    x_ptr,     # (M, K) = (N*H*W, C_in)
    w_ptr,     # (K, N) = (C_in, C_out)  [transposed weight]
    b_ptr,     # (N,) = (C_out,)
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        x = tl.load(x_ptr + offs_m[:, None] * K + k_offs[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + k_offs[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc=acc)
    
    # Add bias and ReLU
    b_mask = offs_n < N
    b = tl.load(b_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + b[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Triton kernel: fused conv2d (3x3, pad=1) + bias + ReLU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'H_in', 'W_in'],
)
@triton.jit
def conv3x3_bias_relu_kernel(
    N_batch, C_in, C_out,
    H_in, W_in,
    H_out, W_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (batch, h_out, w_out)
    w_out_idx = offs_m % W_out
    h_out_idx = (offs_m // W_out) % H_out
    n_idx = offs_m // (H_out * W_out)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over 3x3 kernel and C_in channels
    for kh in range(3):
        for kw in range(3):
            h_in = h_out_idx + kh - 1  # padding=1
            w_in = w_out_idx + kw - 1
            
            valid_hw = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            
            for k in range(0, K, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                
                # Load input: x[n, c_in, h_in, w_in]
                x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K) & valid_hw[:, None]
                x_idx = n_idx[:, None] * C_in * H_in * W_in + offs_k[None, :] * H_in * W_in + h_in[:, None] * W_in + w_in[:, None]
                x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
                
                # Load weight: w[c_out, c_in, kh, kw]
                w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                w_idx = offs_n[None, :] * C_in * 9 + offs_k[:, None] * 9 + kh * 3 + kw
                w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
                
                acc = tl.dot(x, w, acc=acc)
    
    # Add bias and ReLU
    b_mask = offs_n < N
    b = tl.load(b_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + b[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    
    # Store output: out[n, c_out, h_out, w_out]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = n_idx[:, None] * C_out * H_out * W_out + offs_n[None, :] * H_out * W_out + h_out_idx[:, None] * W_out + w_out_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Triton kernel: conv7x7 stride=2 + bias + ReLU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'H_in', 'W_in'],
)
@triton.jit
def conv7x7s2_bias_relu_kernel(
    N_batch, C_in, C_out,
    H_in, W_in,
    H_out, W_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (batch, h_out, w_out)
    w_out_idx = offs_m % W_out
    h_out_idx = (offs_m // W_out) % H_out
    n_idx = offs_m // (H_out * W_out)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            h_in = h_out_idx * 2 + kh  # stride=2, no padding
            w_in = w_out_idx * 2 + kw
            
            valid_hw = (h_in < H_in) & (w_in < W_in)
            
            for k in range(0, K, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                
                x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K) & valid_hw[:, None]
                x_idx = n_idx[:, None] * C_in * H_in * W_in + offs_k[None, :] * H_in * W_in + h_in[:, None] * W_in + w_in[:, None]
                x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
                
                w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                w_idx = offs_n[None, :] * C_in * 49 + offs_k[:, None] * 49 + kh * 7 + kw
                w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
                
                acc = tl.dot(x, w, acc=acc)
    
    b_mask = offs_n < N
    b = tl.load(b_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + b[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = n_idx[:, None] * C_out * H_out * W_out + offs_n[None, :] * H_out * W_out + h_out_idx[:, None] * W_out + w_out_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Triton kernel: adaptive avg pool 2d -> (1,1) + flatten
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_flatten_kernel(
    N, C, HW,  # HW = H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C
    
    n_idx = offsets // C
    c_idx = offsets % C
    
    # Sum over HW
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    base = n_idx * C * HW + c_idx * HW
    
    # Loop over spatial positions
    for hw in range(0, HW, 1):
        v = tl.load(x_ptr + base + hw, mask=mask, other=0.0)
        acc = acc + v
    
    acc = acc / HW.to(tl.float32)
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    """Max pool 2d with kernel=3, stride=2, ceil_mode=True"""
    N, C, H_in, W_in = x.shape
    if ceil_mode:
        H_out = math.ceil((H_in - kernel_size) / stride) + 1
        W_out = math.ceil((W_in - kernel_size) / stride) + 1
    else:
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1
    
    out = torch.empty(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    n_elements = N * C * H_out * W_out
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    max_pool2d_k3s2_kernel[grid](
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        n_elements,
    )
    return out


def triton_conv1x1_bias_relu(x, weight, bias):
    """1x1 conv + bias + relu. x: (N,C_in,H,W), weight: (C_out,C_in,1,1)"""
    N_batch, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Reshape to (N_batch*H*W, C_in) for GEMM
    M = N_batch * H * W
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, C_in)
    w_2d = weight.view(C_out, C_in).t().contiguous()  # (C_in, C_out)
    
    out_2d = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_bias_relu_kernel[grid](
        x_2d, w_2d, bias, out_2d,
        M, C_out, C_in,
    )
    
    # Reshape back to (N_batch, C_out, H, W)
    return out_2d.view(N_batch, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def triton_conv3x3_bias_relu(x, weight, bias):
    """3x3 conv (pad=1) + bias + relu. x: (N,C_in,H,W), weight: (C_out,C_in,3,3)"""
    N_batch, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    H_out, W_out = H_in, W_in  # same size with pad=1
    
    M = N_batch * H_out * W_out
    N = C_out
    K = C_in
    
    out = torch.empty(N_batch, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv3x3_bias_relu_kernel[grid](
        x, weight, bias, out,
        N_batch, C_in, C_out,
        H_in, W_in, H_out, W_out,
        M, N, K,
    )
    return out


def triton_conv7x7s2_bias_relu(x, weight, bias):
    """7x7 conv (stride=2, no pad) + bias + relu."""
    N_batch, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    H_out = (H_in - 7) // 2 + 1
    W_out = (W_in - 7) // 2 + 1
    
    M = N_batch * H_out * W_out
    N = C_out
    K = C_in
    
    out = torch.empty(N_batch, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv7x7s2_bias_relu_kernel[grid](
        x, weight, bias, out,
        N_batch, C_in, C_out,
        H_in, W_in, H_out, W_out,
        M, N, K,
    )
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze -> expand1x1 + expand3x3 -> cat"""
    x = triton_conv1x1_bias_relu(x, sq_w, sq_b)
    e1 = triton_conv1x1_bias_relu(x, ex1_w, ex1_b)
    e3 = triton_conv3x3_bias_relu(x, ex3_w, ex3_b)
    return torch.cat([e1, e3], dim=1)


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_3_squeeze_weight, p_features_3_squeeze_bias,
    p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    p_features_3_expand3x3_weight, p_features_3_expand3x3_bias,
    p_features_4_squeeze_weight, p_features_4_squeeze_bias,
    p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    p_features_4_expand3x3_weight, p_features_4_expand3x3_bias,
    p_features_5_squeeze_weight, p_features_5_squeeze_bias,
    p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
    p_features_5_expand3x3_weight, p_features_5_expand3x3_bias,
    p_features_7_squeeze_weight, p_features_7_squeeze_bias,
    p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
    p_features_7_expand3x3_weight, p_features_7_expand3x3_bias,
    p_features_8_squeeze_weight, p_features_8_squeeze_bias,
    p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
    p_features_8_expand3x3_weight, p_features_8_expand3x3_bias,
    p_features_9_squeeze_weight, p_features_9_squeeze_bias,
    p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
    p_features_9_expand3x3_weight, p_features_9_expand3x3_bias,
    p_features_10_squeeze_weight, p_features_10_squeeze_bias,
    p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
    p_features_10_expand3x3_weight, p_features_10_expand3x3_bias,
    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    p_features_12_expand3x3_weight, p_features_12_expand3x3_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    x,
):
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x = triton_conv7x7s2_bias_relu(x, p_features_0_weight, p_features_0_bias)
    
    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
    
    # features[3]: Fire(96, 16, 64, 64)
    x = fire_module(x,
        p_features_3_squeeze_weight, p_features_3_squeeze_bias,
        p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    
    # features[4]: Fire(128, 16, 64, 64)
    x = fire_module(x,
        p_features_4_squeeze_weight, p_features_4_squeeze_bias,
        p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    
    # features[5]: Fire(128, 32, 128, 128)