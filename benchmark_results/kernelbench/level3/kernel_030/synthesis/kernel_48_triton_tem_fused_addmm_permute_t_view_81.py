import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    stride_m,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):


import torch
import triton
import triton.language as tl
import math

# Fused LayerNorm kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N,
    stride_row,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_row
    out_ptr = out_ptr + row * stride_row

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layernorm_kernel[grid](
        x_2d, weight, bias, out,
        M, N,
        x_2d.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


# Fused GELU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
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
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
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
        triton.Config({'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, stride_row, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_row
    out_ptr = out_ptr + row * stride_row

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Store normalized
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Flatten to 2D along dim
    orig_shape = x.shape
    if dim < 0:
        dim = len(orig_shape) + dim
    # Move dim to last
    if dim != len(orig_shape) - 1:
        x = x.transpose(dim, -1).contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    softmax_kernel[(M,)](x_2d, out, M, N, x_2d.stride(0))
    out = out.reshape(x.shape)
    if dim != len(orig_shape) - 1:
        out = out.transpose(dim, -1).contiguous()
    return out


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
    x, qkv_weight, qkv_bias, proj_weight, proj_bias,
    logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    relative_coords_table, relative_position_index,
    num_heads, window_size, attn_mask=None
):
    B_, N, C = x.shape
    head_dim = C // num_heads

    # Build QKV bias
    q_bias = qkv_bias[:C]
    v_bias = qkv_bias[C:]
    zeros = torch.zeros(C, device=x.device, dtype=x.dtype)
    full_bias = torch.cat([q_bias, zeros, v_bias])

    # QKV projection
    qkv = (x.reshape(B_ * N, C) @ qkv_weight.t() + full_bias).reshape(B_, N, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Normalize q and k
    q_norm = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k_norm = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Scaled cosine attention
    scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
    attn = (q_norm @ k_norm.transpose(-2, -1)) * scale

    # Relative position bias via CPB MLP
    coords = relative_coords_table.reshape(-1, 2)
    cpb = coords @ cpb_mlp_0_weight.t() + cpb_mlp_0_bias
    cpb = torch.relu(cpb)
    cpb = cpb @ cpb_mlp_2_weight.t()
    cpb = cpb.reshape(relative_coords_table.shape[1], relative_coords_table.shape[2], -1)
    cpb = cpb.reshape(-1, num_heads)

    idx = relative_position_index.reshape(-1)
    rel_pos_bias = cpb[idx].reshape(window_size * window_size, window_size * window_size, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16.0
    attn = attn + rel_pos_bias.unsqueeze(0)

    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = triton_softmax(attn, dim=-1)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = x @ proj_weight.t() + proj_bias
    return x


def swin_block(
    x, H, W, window_size, shift_size, num_heads,
    norm1_w, norm1_b, norm2_w, norm2_b,
    qkv_weight, qkv_bias, proj_weight, proj_bias,
    logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    relative_coords_table, relative_position_index,
    fc1_weight, fc1_bias, fc2_weight, fc2_bias,
    attn_mask=None
):
    B, L, C = x.shape
    shortcut = x

    # Norm1
    x = triton_layernorm(x, norm1_w, norm1_b)
    x = x.view(B, H, W, C)

    # Cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

    # Window partition
    x_windows = window_partition(x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Attention
    attn_out = swin_window_attention(
        x_windows, qkv_weight, qkv_bias, proj_weight, proj_bias,
        logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        relative_coords_table, relative_position_index,
    )

    # Reverse window
    attn_out = attn_out.view(-1, window_size, window_size, C)
    x = window_reverse(attn_out, window_size, H, W)

    # Reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

    x = x.view(B, H * W, C)
    x = shortcut + x

    # MLP
    shortcut2 = x
    x = triton_layernorm(x, norm2_w, norm2_b)
    x = x.reshape(B * H * W, C) @ fc1_weight.t() + fc1_bias
    x = triton_gelu(x)
    x = x @ fc2_weight.t() + fc2_bias
    x = x.reshape(B, H * W, C)
    x = shortcut2 + x

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
    x = x @ reduction_weight.t()
    x = triton_layernorm(x, norm_weight, norm_bias)
    return x


def kernel_function(
    p_patch_embed_proj_weight, p_patch_embed_proj_bias,
    p_patch_embed_norm_weight, p_patch_embed_norm_bias,
    p_layers_0_blocks_0_attn_q_bias, p_layers_0_blocks_0_attn_v_bias,
    p_layers_0_blocks_0_attn_qkv_weight, p_layers_0_blocks_0_attn_logit_scale,
    p_layers_0_blocks_0_attn_cpb_mlp_0_weight, p_layers_0_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_0_attn_proj_weight, p_layers_0_blocks_0_attn_proj_bias,
    p_layers_0_blocks_0_norm1_weight, p_layers_0_blocks_0_norm1_bias,
    p_layers_0_blocks_0_mlp_fc1_weight, p_layers_0_blocks_0_mlp_fc1_bias,
    p_layers_0_blocks_0_mlp_fc2_weight, p_layers_0_blocks_0_mlp_fc2_bias,
    p_layers_0_blocks_0_norm2_weight, p_layers_0_blocks_0_norm2_bias,
    p_layers_0_blocks_1_attn_q_bias, p_layers_0_blocks_1_attn_v_bias,
    p_layers_0_blocks_1_attn_qkv_weight, p_layers_0_blocks_1_attn_logit_scale,
    p_layers_0_blocks_1_attn_cpb_mlp_0_weight, p_layers_0_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_0_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_0_blocks_1_attn_proj_weight, p_layers_0_blocks_1_attn_proj_bias,
    p_layers_0_blocks_1_norm1_weight, p_layers_0_blocks_1_norm1_bias,
    p_layers_0_blocks_1_mlp_fc1_weight, p_layers_0_blocks_1_mlp_fc1_bias,
    p_layers_0_blocks_1_mlp_fc2_weight, p_layers_0_blocks_1_mlp_fc2_bias,
    p_layers_0_blocks_1_norm2_weight, p_layers_0_blocks_1_norm2_bias,
    p_layers_0_downsample_reduction_weight,
    p_layers_0_downsample_norm_weight, p_layers_0_downsample_norm_bias,
    p_layers_1_blocks_0_attn_q_bias, p_layers_1_blocks_0_attn_v_bias,
    p_layers_1_blocks_0_attn_qkv_weight, p_layers_1_blocks_0_attn_logit_scale,
    p_layers_1_blocks_0_attn_cpb_mlp_0_weight, p_layers_1_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_0_attn_proj_weight, p_layers_1_blocks_0_attn_proj_bias,
    p_layers_1_blocks_0_norm1_weight, p_layers_1_blocks_0_norm1_bias,
    p_layers_1_blocks_0_mlp_fc1_weight, p_layers_1_blocks_0_mlp_fc1_bias,
    p_layers_1_blocks_0_mlp_fc2_weight, p_layers_1_blocks_0_mlp_fc2_bias,
    p_layers_1_blocks_0_norm2_weight, p_layers_1_blocks_0_norm2_bias,
    p_layers_1_blocks_1_attn_q_bias, p_layers_1_blocks_1_attn_v_bias,
    p_layers_1_blocks_1_attn_qkv_weight, p_layers_1_blocks_1_attn_logit_scale,
    p_layers_1_blocks_1_attn_cpb_mlp_0_weight, p_layers_1_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_1_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_1_blocks_1_attn_proj_weight, p_layers_1_blocks_1_attn_proj_bias,
    p_layers_1_blocks_1_norm1_weight, p_layers_1_blocks_1_norm1_bias,
    p_layers_1_blocks_1_mlp_fc1_weight, p_layers_1_blocks_1_mlp_fc1_bias,
    p_layers_1_blocks_1_mlp_fc2_weight, p_layers_1_blocks_1_mlp_fc2_bias,
    p_layers_1_blocks_1_norm2_weight, p_layers_1_blocks_1_norm2_bias,
    p_layers_1_downsample_reduction_weight,
    p_layers_1_downsample_norm_weight, p_layers_1_downsample_norm_bias,
    p_layers_2_blocks_0_attn_q_bias, p_layers_2_blocks_0_attn_v_bias,
    p_layers_2_blocks_0_attn_qkv_weight, p_layers_2_blocks_0_attn_logit_scale,
    p_layers_2_blocks_0_attn_cpb_mlp_0_weight, p_layers_2_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_0_attn_proj_weight, p_layers_2_blocks_0_attn_proj_bias,
    p_layers_2_blocks_0_norm1_weight, p_layers_2_blocks_0_norm1_bias,
    p_layers_2_blocks_0_mlp_fc1_weight, p_layers_2_blocks_0_mlp_fc1_bias,
    p_layers_2_blocks_0_mlp_fc2_weight, p_layers_2_blocks_0_mlp_fc2_bias,
    p_layers_2_blocks_0_norm2_weight, p_layers_2_blocks_0_norm2_bias,
    p_layers_2_blocks_1_attn_q_bias, p_layers_2_blocks_1_attn_v_bias,
    p_layers_2_blocks_1_attn_qkv_weight, p_layers_2_blocks_1_attn_logit_scale,
    p_layers_2_blocks_1_attn_cpb_mlp_0_weight, p_layers_2_blocks_1_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_1_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_1_attn_proj_weight, p_layers_2_blocks_1_attn_proj_bias,
    p_layers_2_blocks_1_norm1_weight, p_layers_2_blocks_1_norm1_bias,
    p_layers_2_blocks_1_mlp_fc1_weight, p_layers_2_blocks_1_mlp_fc1_bias,
    p_layers_2_blocks_1_mlp_fc2_weight, p_layers_2_blocks_1_mlp_fc2_bias,
    p_layers_2_blocks_1_norm2_weight, p_layers_2_blocks_1_norm2_bias,
    p_layers_2_blocks_2_attn_q_bias, p_layers_2_blocks_2_attn_v_bias,
    p_layers_2_blocks_2_attn_qkv_weight, p_layers_2_blocks_2_attn_logit_scale,
    p_layers_2_blocks_2_attn_cpb_mlp_0_weight, p_layers_2_blocks_2_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_2_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_2_attn_proj_weight, p_layers_2_blocks_2_attn_proj_bias,
    p_layers_2_blocks_2_norm1_weight, p_layers_2_blocks_2_norm1_bias,
    p_layers_2_blocks_2_mlp_fc1_weight, p_layers_2_blocks_2_mlp_fc1_bias,
    p_layers_2_blocks_2_mlp_fc2_weight, p_layers_2_blocks_2_mlp_fc2_bias,
    p_layers_2_blocks_2_norm2_weight, p_layers_2_blocks_2_norm2_bias,
    p_layers_2_blocks_3_attn_q_bias, p_layers_2_blocks_3_attn_v_bias,
    p_layers_2_blocks_3_attn_qkv_weight, p_layers_2_blocks_3_attn_logit_scale,
    p_layers_2_blocks_3_attn_cpb_mlp_0_weight, p_layers_2_blocks_3_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_3_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_3_attn_proj_weight, p_layers_2_blocks_3_attn_proj_bias,
    p_layers_2_blocks_3_norm1_weight, p_layers_2_blocks_3_norm1_bias,
    p_layers_2_blocks_3_mlp_fc1_weight, p_layers_2_blocks_3_mlp_fc1_bias,
    p_layers_2_blocks_3_mlp_fc2_weight, p_layers_2_blocks_3_mlp_fc2_bias,
    p_layers_2_blocks_3_norm2_weight, p_layers_2_blocks_3_norm2_bias,
    p_layers_2_blocks_4_attn_q_bias, p_layers_2_blocks_4_attn_v_bias,
    p_layers_2_blocks_4_attn_qkv_weight, p_layers_2_blocks_4_attn_logit_scale,
    p_layers_2_blocks_4_attn_cpb_mlp_0_weight, p_layers_2_blocks_4_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_4_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_4_attn_proj_weight, p_layers_2_blocks_4_attn_proj_bias,
    p_layers_2_blocks_4_norm1_weight, p_layers_2_blocks_4_norm1_bias,
    p_layers_2_blocks_4_mlp_fc1_weight, p_layers_2_blocks_4_mlp_fc1_bias,
    p_layers_2_blocks_4_mlp_fc2_weight, p_layers_2_blocks_4_mlp_fc2_bias,
    p_layers_2_blocks_4_norm2_weight, p_layers_2_blocks_4_norm2_bias,
    p_layers_2_blocks_5_attn_q_bias, p_layers_2_blocks_5_attn_v_bias,
    p_layers_2_blocks_5_attn_qkv_weight, p_layers_2_blocks_5_attn_logit_scale,
    p_layers_2_blocks_5_attn_cpb_mlp_0_weight, p_layers_2_blocks_5_attn_cpb_mlp_0_bias,
    p_layers_2_blocks_5_attn_cpb_mlp_2_weight,
    p_layers_2_blocks_5_attn_proj_weight, p_layers_2_blocks_5_attn_proj_bias,
    p_layers_2_blocks_5_norm1_weight, p_layers_2_blocks_5_norm1_bias,
    p_layers_2_blocks_5_mlp_fc1_weight, p_layers_2_blocks_5_mlp_fc1_bias,
    p_layers_2_blocks_5_mlp_fc2_weight, p_layers_2_blocks_5_mlp_fc2_bias,
    p_layers_2_blocks_5_norm2_weight, p_layers_2_blocks_5_norm2_bias,
    p_layers_2_downsample_reduction_weight,
    p_layers_2_downsample_norm_weight, p_layers_2_downsample_norm_bias,
    p_layers_3_blocks_0_attn_q_bias, p_layers_3_blocks_0_attn_v_bias,
    p_layers_3_blocks_0_attn_qkv_weight, p_layers_3_blocks_0_attn_logit_scale,
    p_layers_3_blocks_0_attn_cpb_mlp_0_weight, p_layers_3_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_3_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_3_blocks_0_attn_proj_weight, p_layers_3_blocks_0_attn_proj_bias,
    p_layers_3_blocks_0_norm1_weight, p_layers_3_blocks_0_norm1_bias,
    p_layers_3_blocks_0_mlp_fc1_weight, p_layers_3_blocks_0_mlp_fc1_bias,