import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# [Full implementation follows]


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    layer_norm_kernel[(M,)](x_2d, weight, bias, out, M, N, eps, N=N)
    return out.reshape(orig_shape)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N, N=N)
    return out


# ============================================================
# Fused Softmax kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Reshape so last dim is the reduction dim
    orig_shape = x.shape
    if dim != -1 and dim != len(orig_shape) - 1:
        # Move dim to last
        x = x.transpose(dim, -1).contiguous()
        out = torch.empty_like(x)
        x_2d = x.reshape(-1, x.shape[-1])
        M, N = x_2d.shape
        out_2d = out.reshape(-1, N)
        softmax_kernel[(M,)](x_2d, out_2d, M, N, N=N)
        return out.reshape(x.shape).transpose(dim, -1).contiguous().reshape(orig_shape)
    else:
        x_cont = x.contiguous()
        x_2d = x_cont.reshape(-1, x_cont.shape[-1])
        M, N = x_2d.shape
        out_2d = torch.empty_like(x_2d)
        softmax_kernel[(M,)](x_2d, out_2d, M, N, N=N)
        return out_2d.reshape(orig_shape)


# ============================================================
# Fused Linear kernel (using tl.dot)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.float32), mask=out_mask)


def triton_linear(x, weight, bias=None):
    """x: (..., K), weight: (N, K), bias: (N,) -> (..., N)"""
    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight.shape[0]
    x_2d = x.reshape(-1, K)
    M = x_2d.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return out.reshape(orig_shape[:-1] + (N,))


# ============================================================
# Fused Conv2d kernel (implicit GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
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
    s_wco, s_wci, s_wkh, s_wkw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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

        w_ptrs = W_ptr + offs_n[None, :] * s_wco + in_c[:, None] * s_wci + kh[:, None] * s_wkh + kw[:, None] * s_wkw
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    if HAS_BIAS:
        b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Output in NCHW: out[n, c, h, w] = out[n * OUT_H * OUT_W + h * OUT_W + w, c]
    out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    OUT_H = (IN_H + 2 * padding - KERNEL_H) // stride + 1
    OUT_W = (IN_W + 2 * padding - KERNEL_W) // stride + 1

    out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_kernel[grid](
        x, weight, bias if bias is not None else x,
        out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        stride, stride,
        padding, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        HAS_BIAS=(bias is not None),
        BATCH=BATCH, IN_C=IN_C, IN_H=IN_H, IN_W=IN_W,
        OUT_C=OUT_C, OUT_H=OUT_H, OUT_W=OUT_W,
        KERNEL_H=KERNEL_H, KERNEL_W=KERNEL_W,
    )
    return out


# ============================================================
# Swin Transformer V2 implementation using Triton kernels
# ============================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def swin_window_attention(
    x, qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    num_heads, window_size,
    attn_mask=None,
):
    B_, N, C = x.shape
    head_dim = C // num_heads

    # Build qkv bias
    zeros = torch.zeros(C, device=x.device, dtype=x.dtype)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])

    # QKV projection
    qkv = triton_linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Normalize q and k
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)

    # Scaled cosine attention
    scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * scale

    # Relative position bias via CPB MLP
    # cpb_mlp: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
    rct_flat = relative_coords_table.view(-1, 2)
    cpb_hidden = triton_linear(rct_flat, cpb_mlp_0_weight, cpb_mlp_0_bias)
    cpb_hidden = torch.relu(cpb_hidden)
    cpb_out = triton_linear(cpb_hidden, cpb_mlp_2_weight, None)
    cpb_out = cpb_out.view(relative_coords_table.shape[1], relative_coords_table.shape[2], -1)

    # Index relative position bias
    rel_idx = relative_position_index.view(-1)
    cpb_flat = cpb_out.view(-1, num_heads)
    rel_pos_bias = cpb_flat[rel_idx].view(window_size * window_size, window_size * window_size, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16.0
    attn = attn + rel_pos_bias.unsqueeze(0)

    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = triton_softmax(attn, dim=-1)

    x = torch.matmul(attn, v)
    x = x.transpose(1, 2).reshape(B_, N, C)
    x = triton_linear(x, proj_weight, proj_bias)
    return x


def swin_block(
    x, H, W,
    norm1_w, norm1_b,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    norm2_w, norm2_b,
    mlp_fc1_w, mlp_fc1_b,
    mlp_fc2_w, mlp_fc2_b,
    num_heads, window_size,
    shift_size=0,
    attn_mask=None,
):
    B, L, C = x.shape
    shortcut = x

    # Norm1
    x = triton_layer_norm(x, norm1_w, norm1_b)
    x = x.view(B, H, W, C)

    # Cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

    # Window partition
    x_windows = window_partition(x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Attention
    attn_windows = swin_window_attention(
        x_windows, qkv_weight, q_bias, v_bias,
        proj_weight, proj_bias,
        logit_scale,
        cpb_mlp_0_weight, cpb_mlp_0_bias,
        cpb_mlp_2_weight,
        relative_coords_table,
        relative_position_index,
        num_heads, window_size,
        attn_mask=attn_mask,
    )

    # Reverse window
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    x = window_reverse(attn_windows, window_size, H, W)

    # Reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

    x = x.view(B, H * W, C)

    # Residual + Norm2
    x = shortcut + x
    shortcut2 = x
    x = triton_layer_norm(x, norm2_w, norm2_b)

    # MLP
    x = triton_linear(x, mlp_fc1_w, mlp_fc1_b)
    x = triton_gelu(x)
    x = triton_linear(x, mlp_fc2_w, mlp_fc2_b)

    x = shortcut2 + x
    return x


def patch_merging(x, H, W, reduction_weight, norm_w, norm_b):
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)
    x = x.view(B, -1, 4 * C)

    x = triton_linear(x, reduction_weight, None)
    x = triton_layer_norm(x, norm_w, norm_b)
    return x


def kernel_function(
    p_patch_embed_proj_weight,
    p_patch_embed_proj_bias,
    p_patch_embed_norm_weight,
    p_patch_embed_norm_bias,
    p_layers_0_blocks_0_attn_q_bias,
    p_layers_0_blocks_0_attn_v_bias,
    p_layers_0_blocks_0_attn_qkv_weight,
    p_layers_0_blocks_0_attn_logit_scale,
    p_layers_0_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_0_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_0_attn_proj_weight,
    p_layers_0_blocks_0_attn_proj_bias,
    p_layers_0_blocks_0_norm1_weight,
    p_layers_0_blocks_0_norm1_bias,
    p_layers_0_blocks_0_mlp_fc1_weight,
    p_layers_0_blocks_0_mlp_fc1_bias,
    p_layers_0_blocks_0_mlp_fc2_weight,
    p_layers_0_blocks_0_mlp_fc2_bias,
    p_layers_0_blocks_0_norm2_weight,
    p_layers_0_blocks_0_norm2_bias,
    p_layers_0_blocks_1_attn_q_bias,
    p_layers_0_blocks_1_attn_v_bias,
    p_layers_0_blocks_1_attn_qkv_weight,
    p_layers_0_blocks_1_attn_logit_scale,
    p_layers_0_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_0_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_1_attn_proj_weight,
    p_layers_0_blocks_1_attn_proj_bias,
    p_layers_0_blocks_1_norm1_weight,