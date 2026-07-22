import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused bias add + relu kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # bias index = offset % n_channels
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.maximum(result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def apply_bias_relu(x, bias):
    """Apply bias and relu to tensor x with shape (N, C, H, W)"""
    # x is in NCHW format, bias has shape (C,)
    # We need to add bias along channel dimension
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Reshape to make channels the last dim for easy bias indexing
    # Actually, let's work with the flat tensor where channel = (idx // (H*W)) % C
    # But that's complex. Let's use a different approach:
    # Permute to (N, H, W, C), apply bias+relu, permute back
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out = torch.empty_like(x_nhwc)
    
    n_elements = N * H * W * C
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_relu_kernel[grid](
        x_nhwc, bias, out,
        n_elements, C,
    )
    return out.permute(0, 3, 1, 2).contiguous()


# Max pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C,
    IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OUT_H * OUT_W
    mask = offsets < total
    
    # Decode indices: output is in NCHW format
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    oc = (offsets // (OUT_W * OUT_H)) % C
    on = offsets // (OUT_W * OUT_H * C)
    
    # Compute max over kernel window
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            in_idx = on * (C * IN_H * IN_W) + oc * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    N, C, H, W = x.shape
    if ceil_mode:
        OUT_H = (H + stride - 1) // stride  # simplified ceil_mode
        OUT_W = (W + stride - 1) // stride
        # More accurate ceil_mode calculation
        OUT_H = (H - kernel_size + stride - 1 + stride) // stride
        OUT_W = (W - kernel_size + stride - 1 + stride) // stride
        # PyTorch ceil_mode: ceil((H - kernel_size) / stride) + 1
        import math
        OUT_H = math.ceil((H - kernel_size) / stride) + 1
        OUT_W = math.ceil((W - kernel_size) / stride) + 1
    else:
        OUT_H = (H - kernel_size) // stride + 1
        OUT_W = (W - kernel_size) // stride + 1
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C,
        H, W,
        OUT_H, OUT_W,
        kernel_size, kernel_size,
        stride, stride,
        0, 0,
    )
    return out


# Adaptive avg pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    x_ptr,
    out_ptr,
    N, C,
    IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C
    mask = offsets < total
    
    oc = offsets % C
    on = offsets // C
    
    # Sum over all spatial positions
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IN_H * IN_W
    
    for h in range(IN_H):
        for w in range(IN_W):
            in_idx = on * (C * IN_H * IN_W) + oc * (IN_H * IN_W) + h * IN_W + w
            val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
            acc = acc + val
    
    avg = acc / count
    tl.store(out_ptr + offsets, avg, mask=mask)


# Implicit GEMM conv2d kernel for 1x1 convolutions
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_relu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)
    
    # Add bias and relu
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def conv1x1_bias_relu(x, weight, bias):
    """
    returns: (N, C_out, H, W) after relu
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Reshape to (N*H*W, C_in) for GEMM
    x_flat = x.permute(0, 2, 3, 1).contiguous().reshape(N * H * W, C_in)
    w_flat = weight.reshape(C_out, C_in)
    
    M = N * H * W
    K = C_in
    out_flat = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_bias_relu_kernel[grid](
        x_flat, w_flat, bias, out_flat,
        M, C_out, K,
    )
    
    return out_flat.reshape(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


# 3x3 conv with bias and relu using implicit GEMM
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3x3_bias_relu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N_batch, C_in, IN_H, IN_W,
    C_out, OUT_H, OUT_W,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # M = N_batch * OUT_H * OUT_W
    # K = C_in * 3 * 3
    # N = C_out
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode k into (c_in, kh, kw)
        c_in = offs_k % C_in
        kh = (offs_k // C_in) % 3
        kw = offs_k // (C_in * 3)
        
        # Decode m into (n, oh, ow)
        ow = offs_m % OUT_W
        oh = (offs_m // OUT_W) % OUT_H
        n = offs_m // (OUT_W * OUT_H)
        
        # Input coordinates
        ih = oh[:, None] * 1 + kh[None, :] - 1  # stride=1, pad=1
        iw = ow[:, None] * 1 + kw[None, :] - 1
        
        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        x_mask = valid_m[:, None] & valid_k[None, :] & valid_spatial
        x_idx = n[:, None] * (C_in * IN_H * IN_W) + c_in[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
        
        # Weight: (C_out, C_in, 3, 3) -> index as (c_out, c_in, kh, kw)
        # w_idx for each (k, n): w[offs_n, c_in, kh, kw]
        w_mask = valid_k[:, None] & (offs_n[None, :] < N)
        w_idx = offs_n[None, :] * (C_in * 9) + c_in[:, None] * 9 + kh[:, None] * 3 + kw[:, None]
        w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias and relu
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    
    # Store: output is (N_batch, C_out, OUT_H, OUT_W) in NCHW
    # offs_m -> (n, oh, ow), offs_n -> c_out
    ow_out = offs_m % OUT_W
    oh_out = (offs_m // OUT_W) % OUT_H
    n_out = offs_m // (OUT_W * OUT_H)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = n_out[:, None] * (C_out * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh_out[:, None] * OUT_W + ow_out[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def conv3x3_bias_relu(x, weight, bias):
    """
    padding=1, stride=1
    returns: (N, C_out, H, W) after relu
    """
    N_batch, C_in, IN_H, IN_W = x.shape
    C_out = weight.shape[0]
    OUT_H = IN_H  # padding=1, stride=1
    OUT_W = IN_W
    
    M = N_batch * OUT_H * OUT_W
    N = C_out
    K = C_in * 9
    
    out = torch.empty(N_batch, C_out, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv3x3_bias_relu_kernel[grid](
        x, weight, bias, out,
        N_batch, C_in, IN_H, IN_W,
        C_out, OUT_H, OUT_W,
        M, N, K,
    )
    
    return out


# Large 7x7 conv kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv7x7_s2_bias_relu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N_batch, C_in, IN_H, IN_W,
    C_out, OUT_H, OUT_W,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # M = N_batch * OUT_H * OUT_W
    # K = C_in * 7 * 7
    # N = C_out
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode k into (c_in, kh, kw) - weight layout (C_out, C_in, 7, 7)
        # k = kw + 7 * kh + 49 * c_in
        c_in = offs_k // 49
        kh = (offs_k % 49) // 7
        kw = offs_k % 7
        
        # Decode m into (n, oh, ow)
        ow = offs_m % OUT_W
        oh = (offs_m // OUT_W) % OUT_H
        n = offs_m // (OUT_W * OUT_H)
        
        # Input coordinates: stride=2, no padding
        ih = oh[:, None] * 2 + kh[None, :]
        iw = ow[:, None] * 2 + kw[None, :]
        
        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        x_mask = valid_m[:, None] & valid_k[None, :] & valid_spatial
        x_idx = n[:, None] * (C_in * IN_H * IN_W) + c_in[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
        
        # Weight: (C_out, C_in, 7, 7)
        # w_idx for each (k, n): w[offs_n, c_in, kh, kw]
        w_mask = valid_k[:, None] & (offs_n[None, :] < N)
        w_idx = offs_n[None, :] * (C_in * 49) + c_in[:, None] * 49 + kh[:, None] * 7 + kw[:, None]
        w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias and relu
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    
    # Store: output is (N_batch, C_out, OUT_H, OUT_W) in NCHW
    ow_out = offs_m % OUT_W
    oh_out = (offs_m // OUT_W) % OUT_H
    n_out = offs_m // (OUT_W * OUT_H)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = n_out[:, None] * (C_out * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh_out[:, None] * OUT_W + ow_out[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def conv7x7_s2_bias_relu(x, weight, bias):
    """
    stride=2, no padding
    returns: (N, C_out, OUT_H, OUT_W) after relu
    """
    N_batch, C_in, IN_H, IN_W = x.shape
    C_out = weight.shape[0]
    OUT_H = (IN_H - 7) // 2 + 1
    OUT_W = (IN_W - 7) // 2 + 1
    
    M = N_batch * OUT_H * OUT_W
    N = C_out
    K = C_in * 49
    
    out = torch.empty(N_batch, C_out, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv7x7_s2_bias_relu_kernel[grid](
        x, weight, bias, out,
        N_batch, C_in, IN_H, IN_W,
        C_out, OUT_H, OUT_W,
        M, N, K,
    )
    
    return out


# Adaptive avg pool 2d for output 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 64}, num_warps=4),
        triton.Config({'BLOCK_NC': 128}, num_warps=4),
        triton.Config({'BLOCK_NC': 32}, num_warps=2),
        triton.Config({'BLOCK_NC': 256}, num_warps=8),
    ],
    key=['N_batch', 'C', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_v2_kernel(
    x_ptr,
    out_ptr,
    N_batch, C,
    IN_H, IN_W,
    BLOCK_NC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    total = N_batch * C
    mask = offs < total
    
    n = offs // C
    c = offs % C
    
    acc = tl.zeros([BLOCK_NC], dtype=tl.float32)
    count = IN_H * IN_W
    
    for h in range(IN_H):
        for w in range(IN_W):
            idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + h * IN_W + w
            val = tl.load(x_ptr + idx, mask=mask, other=0.0)
            acc = acc + val
    
    avg = acc / count
    tl.store(out_ptr + offs, avg, mask=mask)


def fire_module_triton(x,
                        sq_w, sq_b,