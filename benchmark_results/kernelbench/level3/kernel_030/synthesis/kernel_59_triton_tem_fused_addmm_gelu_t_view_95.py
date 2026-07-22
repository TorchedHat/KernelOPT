import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_R': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_R': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_R': 64}, num_warps=2, num_stages=3),
    ],
    key=['N_ROWS', 'N_COLS'],
)
@triton.jit
def fused_layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N_ROWS, N_COLS,
    stride_row,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_N
    row_offs = row_start + tl.arange(0, BLOCK_N)
    row_mask = row_offs < N_ROWS

    # Compute mean
    mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    for col_off in range(0, N_COLS, BLOCK_R):
        col_offs = col_off + tl.arange(0, BLOCK_R)
        col_mask = col_offs < N_COLS
        ptrs = x_ptr + row_offs[:, None] * stride_row + col_offs[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=1)
    mean = mean / N_COLS

    # Compute variance
    var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for col_off in range(0, N_COLS, BLOCK_R):
        col_offs = col_off + tl.arange(0, BLOCK_R)
        col_mask = col_offs < N_COLS
        ptrs = x_ptr + row_offs[:, None] * stride_row + col_offs[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean[:, None]
        diff = tl.where(mask, diff, 0.0)
        var += tl.sum(diff * diff, axis=1)
    var = var / N_COLS
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for col_off in range(0, N_COLS, BLOCK_R):
        col_offs = col_off + tl.arange(0, BLOCK_R)
        col_mask = col_offs < N_COLS
        ptrs = x_ptr + row_offs[:, None] * stride_row + col_offs[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offs, mask=col_mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
        x_norm = (x - mean[:, None]) * rstd[:, None]
        out = x_norm * w[None, :] + b[None, :]
        out_ptrs = out_ptr + row_offs[:, None] * stride_row + col_offs[None, :]
        tl.store(out_ptrs, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N_ROWS, N_COLS = x_2d.shape
    out = torch.empty_like(x_2d)
    stride_row = x_2d.stride(0)
    
    BLOCK_N = 16
    grid = lambda meta: (triton.cdiv(N_ROWS, meta['BLOCK_N']),)
    fused_layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        N_ROWS, N_COLS,
        stride_row,
        eps,
    )
    return out.reshape(orig_shape)


# Fused GEMM kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def triton_linear(x, weight, bias=None):
    """x: (..., K), weight: (N, K) -> (..., N)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    matmul_kernel[grid](
        x_2d, weight, out, bias if bias is not None else x_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )
    
    return out.reshape(*orig_shape[:-1], N)


# Fused GELU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_val = libdevice.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# Fused softmax kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    ],
    key=['N_COLS'],
)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, N_ROWS, N_COLS, stride_row, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    if row_idx >= N_ROWS:
        return
    
    row_start = row_idx * stride_row
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load row
    x = tl.load(x_ptr + row_start + cols, mask=cols < N_COLS, other=float('-inf')).to(tl.float32)
    
    # Compute softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    
    tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=cols < N_COLS)


def triton_softmax(x, dim=-1):
    assert dim == -1 or dim == x.dim() - 1
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N_ROWS, N_COLS = x_2d.shape
    out = torch.empty_like(x_2d)
    
    # Need BLOCK_SIZE >= N_COLS, power of 2
    BLOCK_SIZE = 1
    while BLOCK_SIZE < N_COLS:
        BLOCK_SIZE *= 2
    
    grid = (N_ROWS,)
    softmax_kernel[grid](x_2d, out, N_ROWS, N_COLS, x_2d.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(orig_shape)


# Conv2d kernel (implicit GEMM)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    s_xn, s_xc, s_xh, s_xw,
    s_wo, s_wi, s_wkh, s_wkw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)
        
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K) &
            (offs_m[:, None] < M_TOTAL)
        )
        
        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        w_ptrs = W_ptr + offs_n[None, :] * s_wo + in_c[:, None] * s_wi + kh[:, None] * s_wkh + kw[:, None] * s_wkw
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + batch[:, None] * (OUT_H * OUT_W * OUT_C) + out_h[:, None] * (OUT_W * OUT_C) + out_w[:, None] * OUT_C + offs_n[None, :]
    mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    """x: (N, C, H, W), weight: (Co, Ci, R, S)"""
    N, Ci, H, W = x.shape
    Co, _, R, S = weight.shape
    
    OUT_H = (H + 2 * padding - R) // stride + 1
    OUT_W = (W + 2 * padding - S) // stride + 1
    
    # Output in NHWC format
    out = torch.empty((N, OUT_H, OUT_W, Co), device=x.device, dtype=x.dtype)
    
    M_TOTAL = N * OUT_H * OUT_W
    
    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(Co, meta['BLOCK_N']))
    
    conv2d_kernel[grid](
        x, weight, bias if bias is not None else x, out,
        N, Ci, H, W,
        Co, OUT_H, OUT_W,
        R, S,
        stride, stride,
        padding, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        HAS_BIAS=(bias is not None),
        BATCH=N, IN_C=Ci, OUT_C=Co, OUT_H=OUT_H, OUT_W=OUT_W,
    )
    
    # Convert NHWC -> NCHW
    return out.permute(0, 3, 1, 2).contiguous()


# Sigmoid kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, N)
    return out


# ReLU inplace kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def relu_inplace_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(x_ptr + offs, out.to(tl.float32), mask=mask)


def triton_relu_inplace(x):
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_inplace_kernel[grid](x, N)
    return x


# Vector norm kernel (L2 norm along last dim)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16, 'BLOCK_R': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 8, 'BLOCK_R': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_R': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 4, 'BLOCK_R': 128}, num_warps=8, num_stages=2),
    ],
    key=['N_ROWS', 'N_COLS'],
)
@triton.jit
def vector_norm_kernel(
    x_ptr, out_ptr,
    N_ROWS, N_COLS,
    stride_row,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_N
    row_offs = row_start + tl.arange(0, BLOCK_N)
    row_mask = row_offs < N_ROWS
    
    norm_sq = tl.zeros([BLOCK_N], dtype=tl.float32)
    for col_off in range(0, N_COLS, BLOCK_R):
        col_offs = col_off + tl.arange(0, BLOCK_R)
        col_mask = col_offs < N_COLS
        ptrs = x_ptr + row_offs[:, None] * stride_row + col_offs[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        norm_sq += tl.sum(x * x, axis=1)
    
    norm = tl.sqrt(norm_sq)
    out_ptrs = out_ptr + row_offs
    tl.store(out_ptrs, norm.to(tl.float32), mask=row_mask)


def triton_vector_norm(x, dim=-1, keepdim=True):
    """Compute L2 norm along last dimension"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N_ROWS, N_COLS = x_2d.shape
    out = torch.empty(N_ROWS, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(N_ROWS, meta['BLOCK_N']),)
    vector_norm_kernel[grid](
        x_2d, out,
        N_ROWS, N_COLS,
        x_2d.stride(0),
    )
    
    if keepdim:
        return out.reshape(*orig_shape[:-1], 1)
    return out.reshape(*orig_shape[:-1])


def window_partition_triton(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_triton(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def swin_attention(
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    relative_coords_table,
    relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    num_heads,
    window_size=7,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads
    
    # Build QKV bias
    zeros = torch.zeros(C, device=q_bias.device, dtype=q_bias.dtype)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])
    
    # QKV projection
    qkv = triton_linear(x_windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # Wait, reshape should be (B_, N, 3, num_heads, head_dim)
    # permute to (3, B_, num_heads, N, head_dim)
    qkv = x_windows @ qkv_weight.T + qkv_bias
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    # Normalize q and k
    q_norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp(min=1e-6)
    q = q / q_norm
    k_norm = torch.linalg.vector_norm(k, dim=-1, keepdim=True).clamp(min=1e-6)
    k = k / k_norm
    
    # Attention scores
    attn = q @ k.transpose(-2, -1)