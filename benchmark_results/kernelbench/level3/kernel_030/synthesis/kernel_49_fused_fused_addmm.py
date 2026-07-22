import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_D': 64}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def fused_layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, D,
    stride_n,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row < N
    
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for d_off in range(0, D, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)
        d_mask = d < D
        x = tl.load(x_ptr + row[:, None] * stride_n + d[None, :],
                    mask=row_mask[:, None] & d_mask[None, :], other=0.0)
        x = x.to(tl.float32)
        mean += tl.sum(x, axis=1)
    mean = mean / D
    
    for d_off in range(0, D, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)
        d_mask = d < D
        x = tl.load(x_ptr + row[:, None] * stride_n + d[None, :],
                    mask=row_mask[:, None] & d_mask[None, :], other=0.0)
        x = x.to(tl.float32)
        diff = x - mean[:, None]
        var += tl.sum(diff * diff, axis=1)
    var = var / D
    rstd = 1.0 / tl.sqrt(var + eps)
    
    for d_off in range(0, D, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)
        d_mask = d < D
        x = tl.load(x_ptr + row[:, None] * stride_n + d[None, :],
                    mask=row_mask[:, None] & d_mask[None, :], other=0.0)
        x = x.to(tl.float32)
        w = tl.load(weight_ptr + d, mask=d_mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        out = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
        tl.store(out_ptr + row[:, None] * stride_n + d[None, :],
                 out.to(tl.float32),
                 mask=row_mask[:, None] & d_mask[None, :])


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
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 1024}, num_warps=16),
    ],
    key=['D'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, D,
    stride_n,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    
    # Compute mean
    mean = tl.zeros([], dtype=tl.float32)
    for d_off in range(0, D, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(x_ptr + row * stride_n + d, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / D
    
    # Compute variance
    var = tl.zeros([], dtype=tl.float32)
    for d_off in range(0, D, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(x_ptr + row * stride_n + d, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / D
    rstd = 1.0 / libdevice.sqrt(var + eps)
    
    # Normalize and store
    for d_off in range(0, D, BLOCK_D):
        d = d_off + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(x_ptr + row * stride_n + d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + d, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + d, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row * stride_n + d, out, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    layer_norm_kernel[(N,)](
        x_2d, weight, bias, out,
        N, D,
        x_2d.stride(0),
        eps,
    )
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
    out = x * cdf
    tl.store(out_ptr + offs, out, mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    gelu_kernel[(triton.cdiv(N, 1024),)](x, out, N)
    return out


# ============================================================
# Fused softmax kernel
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
def softmax_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    tl.store(out_ptr + row * stride_m + offs, out, mask=mask)


def triton_softmax(x, dim=-1):
    # Only handle last dim softmax
    assert dim == -1 or dim == x.dim() - 1
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N)
    M = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    softmax_kernel[(M,)](x_2d, out, M, N, x_2d.stride(0))
    return out.reshape(orig_shape)


# ============================================================
# Main model forward using Triton kernels for fusable ops
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


def swin_attention(
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
    qkv = torch.nn.functional.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv shape: [3, B_, num_heads, N, head_dim] -- wait, let me check
    # Actually: reshape to [B_, N, 3, num_heads, head_dim], permute to [3, B_, num_heads, N, head_dim]
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

    # Cosine attention
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)

    # Logit scale
    scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * scale

    # Relative position bias via CPB MLP
    # cpb_mlp: linear(2->512, relu) -> linear(512->num_heads)
    rel_pos_bias_table = torch.nn.functional.linear(
        relative_coords_table.reshape(-1, 2), cpb_mlp_0_weight, cpb_mlp_0_bias
    )
    rel_pos_bias_table = torch.nn.functional.relu(rel_pos_bias_table)
    rel_pos_bias_table = torch.nn.functional.linear(rel_pos_bias_table, cpb_mlp_2_weight)
    rel_pos_bias_table = rel_pos_bias_table.view(-1, num_heads)

    # Index into table
    idx = relative_position_index.view(-1)
    rel_pos_bias = rel_pos_bias_table[idx].view(window_size * window_size, window_size * window_size, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16.0
    attn = attn + rel_pos_bias.unsqueeze(0)

    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = triton_softmax(attn, dim=-1)
    # In inference, dropout is identity
    x = torch.matmul(attn, v)
    x = x.transpose(1, 2).reshape(B_, N, C)
    x = torch.nn.functional.linear(x, proj_weight, proj_bias)
    return x


def swin_block(
    x, H, W,
    norm1_weight, norm1_bias,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    norm2_weight, norm2_bias,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    num_heads, window_size,
    shift_size=0,
    attn_mask=None,
):
    B, L, C = x.shape
    shortcut = x

    # Norm1
    x_norm = triton_layer_norm(x, norm1_weight, norm1_bias)

    x_norm = x_norm.view(B, H, W, C)

    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x_norm, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x_norm

    # Window partition
    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Attention
    attn_windows = swin_attention(
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
    shifted_x = window_reverse(attn_windows, window_size, H, W)

    # Reverse cyclic shift
    if shift_size > 0:
        x_out = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x_out = shifted_x

    x_out = x_out.view(B, H * W, C)
    x = shortcut + x_out

    # MLP
    x_norm2 = triton_layer_norm(x, norm2_weight, norm2_bias)
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