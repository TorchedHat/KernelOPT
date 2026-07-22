import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch.nn.functional as F
import math

# Fused LayerNorm kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(


import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)

    mean = mean / N

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)

    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_ptr + row * stride_ym + cols * stride_yn, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    fused_layer_norm_kernel[grid](
        x_2d, weight, bias, y,
        M, N,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
        eps,
    )
    return y.reshape(orig_shape)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# ============================================================
# Fused Softmax kernel (row-wise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Normalize
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_ptr + row * N + cols, y.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Only handle last dim softmax
    assert dim == -1 or dim == x.dim() - 1
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, y, M, N)
    return y.reshape(orig_shape)


# ============================================================
# SwinV2 model implementation using Triton kernels
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
    qkv_weight, qkv_bias,
    proj_weight, proj_bias,
    logit_scale,
    relative_coords_table,
    relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    num_heads,
    attn_mask=None,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads

    # QKV projection
    qkv = torch.nn.functional.linear(x_windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Cosine attention
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)

    logit_scale_clamped = torch.clamp(logit_scale, max=math.log(100.0)).exp()
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * logit_scale_clamped

    # Relative position bias via CPB MLP
    # cpb_mlp: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
    rel_pos_bias_table = torch.nn.functional.linear(relative_coords_table, cpb_mlp_0_weight, cpb_mlp_0_bias)
    rel_pos_bias_table = torch.nn.functional.relu(rel_pos_bias_table)
    rel_pos_bias_table = torch.nn.functional.linear(rel_pos_bias_table, cpb_mlp_2_weight)
    rel_pos_bias_table = rel_pos_bias_table.view(-1, num_heads)

    rel_pos_idx = relative_position_index.view(-1)
    rel_pos_bias = rel_pos_bias_table[rel_pos_idx].view(N, N, num_heads)
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
    x = torch.nn.functional.linear(x, proj_weight, proj_bias)
    return x


def swin_block(
    norm1_w, norm1_b,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    relative_coords_table,
    relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    norm2_w, norm2_b,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    num_heads,
    window_size,
    shift_size,
    H, W,
    attn_mask=None,
):
    B, L, C = x.shape
    shortcut = x

    # Norm1
    x_norm = triton_layer_norm(x, norm1_w, norm1_b)
    x_norm = x_norm.view(B, H, W, C)

    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x_norm, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x_norm

    # Window partition
    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Build qkv_bias with zeros for k
    zeros_k = torch.zeros_like(q_bias)
    qkv_bias = torch.cat([q_bias, zeros_k, v_bias])

    # Attention
    attn_windows = swin_window_attention(
        x_windows, qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        logit_scale,
        relative_coords_table, relative_position_index,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        num_heads, attn_mask,
    )

    # Window reverse
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)

    # Reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x

    x = x.view(B, H * W, C)

    # Residual
    x = shortcut + x

    # Norm2 + MLP
    x_norm2 = triton_layer_norm(x, norm2_w, norm2_b)
    # MLP
    x_mlp = torch.nn.functional.linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    x_mlp = triton_gelu(x_mlp)
    x_mlp = torch.nn.functional.linear(x_mlp, mlp_fc2_weight, mlp_fc2_bias)

    x = x + x_mlp
    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)
    x = x.view(B, -1, 4 * C)

    x = torch.nn.functional.linear(x, reduction_weight)
    x = triton_layer_norm(x, norm_weight, norm_bias)
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
    p_layers_0_blocks_1_norm1_bias,
    p_layers_0_blocks_1_mlp_fc1_weight,
    p_layers_0_blocks_1_mlp_fc1_bias,
    p_layers_0_blocks_1_mlp_fc2_weight,
    p_layers_0_blocks_1_mlp_fc2_bias,
    p_layers_0_blocks_1_norm2_weight,
    p_layers_0_blocks_1_norm2_bias,
    p_layers_0_downsample_reduction_weight,
    p_layers_0_downsample_norm_weight,
    p_layers_0_downsample_norm_bias,
    p_layers_1_blocks_0_attn_q_bias,
    p_layers_1_blocks_0_attn_v_bias,
    p_layers_1_blocks_0_attn_qkv_weight,
    p_layers_1_blocks_0_attn_logit_scale,
    p_layers_1_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_1_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_0_attn_proj_weight,
    p_layers_1_blocks_0_attn_proj_bias,
    p_layers_1_blocks_0_norm1_weight,
    p_layers_1_blocks_0_norm1_bias,
    p_layers_1_blocks_0_mlp_fc1_weight,
    p_layers_1_blocks_0_mlp_fc1_bias,
    p_layers_1_blocks_0_mlp_fc2_weight,
    p_layers_1_blocks_0_mlp_fc2_bias,
    p_layers_1_blocks_0_norm2_weight,
    p_layers_1_blocks_0_norm2_bias,
    p_layers_1_blocks_1_attn_q_bias,
    p_layers_1_blocks_1_attn_v_bias,
    p_layers_1_blocks_1_attn_qkv_weight,
    p_layers_1_blocks_1_attn_logit_scale,
    p_layers_1_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_1_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_1_attn_proj_weight,
    p_layers_1_blocks_1_attn_proj_bias,
    p_layers_1_blocks_1_norm1_weight,
    p_layers_1_blocks_1_norm1_bias,
    p_layers_1_blocks_1_mlp_fc1_weight,
    p_layers_1_blocks_1_mlp_fc1_bias,
    p_layers_1_blocks_1_mlp_fc2_weight,
    p_layers_1_blocks_1_mlp_fc2_bias,
    p_layers_1_blocks_1_norm2_weight,
    p_layers_1_blocks_1_norm2_bias,
    p_layers_1_downsample_reduction_weight,
    p_layers_1_downsample_norm_weight,
    p_layers_1_downsample_norm_bias,
    p_layers_2_blocks_0_attn_q_bias,
    p_layers_2_blocks_0_attn_v_bias,
    p_layers_2_blocks_0_attn_qkv_weight,
    p_layers_2_blocks_0_attn_logit_scale,
    p_layers_2_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_0_attn_proj_weight,
    p_layers_2_blocks_0_attn_proj_bias,
    p_layers_2_blocks_0_norm1_weight,
    p_layers_2_blocks_0_norm1_bias,
    p_layers_2_blocks_0_mlp_fc1_weight,
    p_layers_2_blocks_0_mlp_fc1_bias,
    p_layers_2_blocks_0_mlp_fc2_weight,
    p_layers_2_blocks_0_mlp_fc2_bias,
    p_layers_2_blocks_0_norm2_weight,
    p_layers_2_blocks_0_norm2_bias,
    p_layers_2_blocks_1_attn_q_bias,
    p_layers_2_blocks_1_attn_v_bias,
    p_layers_2_blocks_1_attn_qkv_weight,
    p_layers_2_blocks_1_attn_logit_scale,
    p_layers_2_blocks_1_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_1_attn_proj_weight,
    p_layers_2_blocks_1_attn_proj_bias,
    p_layers_2_blocks_1_norm1_weight,
    p_layers_2_blocks_1_norm1_bias,
    p_layers_2_blocks_1_mlp_fc1_weight,
    p_layers_2_blocks_1_mlp_fc1_bias,
    p_layers_2_blocks_1_mlp_fc2_weight,
    p_layers_2_blocks_1_mlp_fc2_bias,
    p_layers_2_blocks_1_norm2_weight,
    p_layers_2_blocks_1_norm2_bias,
    p_layers_2_blocks_2_attn_q_bias,
    p_layers_2_blocks_2_attn_v_bias,
    p_layers_2_blocks_2_attn_qkv_weight,
    p_layers_2_blocks_2_attn_logit_scale,
    p_layers_2_blocks_2_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_2_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_2_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_2_attn_proj_weight,
    p_layers_2_blocks_2_attn_proj_bias,
    p_layers_2_blocks_2_norm1_weight,
    p_layers_2_blocks_2_norm1_bias,
    p_layers_2_blocks_2_mlp_fc1_weight,
    p_layers_2_blocks_2_mlp_fc1_bias,
    p_layers_2_blocks_2_mlp_fc2_weight,
    p_layers_2_blocks_2_mlp_fc2_bias,
    p_layers_2_blocks_2_norm2_weight,
    p_layers_2_blocks_2_norm2_bias,
    p_layers_2_blocks_3_attn_q_bias,
    p_layers_2_blocks_3_attn_v_bias,
    p_layers_2_blocks_3_attn_qkv_weight,
    p_layers_2_blocks_3_attn_logit_scale,
    p_layers_2_blocks_3_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_3_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_3_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_3_attn_proj_weight,
    p_layers_2_blocks_3_attn_proj_bias,
    p_layers_2_blocks_3_norm1_weight,
    p_layers_2_blocks_3_norm1_bias,
    p_layers_2_blocks_3_mlp_fc1_weight,
    p_layers_2_blocks_3_mlp_fc1_bias,
    p_layers_2_blocks_3_mlp_fc2_weight,
    p_layers_2_blocks_3_mlp_fc2_bias,
    p_layers_2_blocks_3_norm2_weight,
    p_layers_2_blocks_3_norm2_bias,
    p_layers_2_blocks_4_attn_q_bias,
    p_layers_2_blocks_4_attn_v_bias,
    p_layers_2_blocks_4_attn_qkv_weight,
    p_layers_2_blocks_4_attn_logit_scale,
    p_layers_2_blocks_4_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_4_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_4_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_4_attn_proj_weight,
    p_layers_2_blocks_4_attn_proj_bias,
    p_layers_2_blocks_4_norm1_weight,
    p_layers_2_blocks_4_norm1_bias,
    p_layers_2_blocks_4_mlp_fc1_weight,
    p_layers_2_blocks_4_mlp_fc1_bias,
    p_layers_2_blocks_4_mlp_fc2_weight,
    p_layers_2_blocks_4_mlp_fc2_bias,
    p_layers_2_blocks_4_norm2_weight,
    p_layers_2_blocks_4_norm2_bias,
    p_layers_2_blocks_5_attn_q_bias,
    p_layers_2_blocks_5_attn_v_bias,
    p_layers_2_blocks_5_attn_qkv_weight,
    p_layers_2_blocks_5_attn_logit_scale,
    p_layers_2_blocks_5_attn_cpb_mlp_0_weight,
    p_layers_2_blocks_5_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_5_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_5_attn_proj_weight,
    p_layers_2_blocks_5_attn_proj_bias,
    p_layers_2_blocks_5_norm1_weight,
    p_layers_2_blocks_5_norm1_bias,
    p_layers_2_blocks_5_mlp_fc1_weight,