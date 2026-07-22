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
    M, N,
    stride_m,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_m + c, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x * mask.to(tl.float32), axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_m + c, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = libdevice.rsqrt(var + eps)
    
    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_m + c, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row * stride_m + c, out.to(tl.float32), mask=mask)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
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
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


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
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_m + c, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_m + c, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max) * mask.to(tl.float32), axis=0)
    
    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_m + c, mask=mask, other=float('-inf')).to(tl.float32)
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_ptr + row * stride_m + c, out.to(tl.float32), mask=mask)


# ============================================================
# Fused L2-normalize kernel (for cosine attention)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def l2norm_kernel(x_ptr, out_ptr, total_vecs, D, stride_d, BLOCK_SIZE: tl.constexpr):
    """Normalize each vector of length D along last dim."""
    vid = tl.program_id(0)
    if vid >= total_vecs:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute norm
    norm_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_SIZE):
        c = off + cols
        mask = c < D
        x = tl.load(x_ptr + vid * stride_d + c, mask=mask, other=0.0).to(tl.float32)
        norm_sq += tl.sum(x * x * mask.to(tl.float32), axis=0)
    norm = libdevice.sqrt(norm_sq)
    norm = tl.maximum(norm, 1e-12)
    
    # Normalize
    for off in range(0, D, BLOCK_SIZE):
        c = off + cols
        mask = c < D
        x = tl.load(x_ptr + vid * stride_d + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + vid * stride_d + c, (x / norm).to(tl.float32), mask=mask)


# ============================================================
# Fused sigmoid + scale kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_scale_kernel(x_ptr, out_ptr, scale, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (1.0 / (1.0 + tl.exp(-x))) * scale
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    """Apply layer norm using Triton kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    layer_norm_kernel[(M,)](x_2d, weight, bias, out, M, N, N, eps)
    return out.reshape(orig_shape)


def triton_gelu(x):
    """Apply GELU using Triton kernel."""
    out = torch.empty_like(x)
    N = x.numel()
    gelu_kernel[(triton.cdiv(N, 512),)](x, out, N)
    return out


def triton_softmax(x, dim=-1):
    """Apply softmax along last dimension using Triton kernel."""
    assert dim == -1 or dim == x.dim() - 1
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    softmax_kernel[(M,)](x_2d, out, M, N, N)
    return out.reshape(orig_shape)


def triton_l2norm(x):
    """L2 normalize along last dimension."""
    orig_shape = x.shape
    D = orig_shape[-1]
    total_vecs = x.numel() // D
    x_flat = x.reshape(total_vecs, D)
    out = torch.empty_like(x_flat)
    l2norm_kernel[(total_vecs,)](x_flat, out, total_vecs, D, D)
    return out.reshape(orig_shape)


# ============================================================
# Main Swin Transformer V2 implementation
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
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    num_heads,
    attn_mask=None,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads

    # QKV projection
    qkv = torch.nn.functional.linear(x_windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: [3, B_, num_heads, N, head_dim] -> wait, let me check
    # Actually reshape to [B_, N, 3, num_heads, head_dim] then permute(2,0,3,1,4)
    qkv = x_windows @ qkv_weight.T + qkv_bias
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

    # Cosine attention: normalize q and k
    q = triton_l2norm(q)
    k = triton_l2norm(k)

    # Logit scale
    scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()  # [num_heads, 1, 1]

    # Attention scores
    attn = (q @ k.transpose(-2, -1)) * scale  # [B_, num_heads, N, N]

    # Relative position bias via CPB MLP
    # relative_coords_table: [1, 2*Wh-1, 2*Ww-1, 2]
    rct = relative_coords_table.view(-1, 2)  # [169, 2]
    # MLP: linear -> relu -> linear
    cpb = torch.nn.functional.linear(rct, cpb_mlp_0_weight, cpb_mlp_0_bias)
    cpb = torch.nn.functional.relu(cpb)
    cpb = torch.nn.functional.linear(cpb, cpb_mlp_2_weight)  # [169, num_heads]
    cpb = cpb.view(1, 13, 13, num_heads)  # [1, 13, 13, num_heads]
    
    # Index relative position bias
    rpi = relative_position_index.view(-1)  # [N*N]
    cpb_flat = cpb.view(-1, num_heads)  # [169, num_heads]
    rel_pos_bias = cpb_flat[rpi].view(N, N, num_heads)  # [N, N, num_heads]
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16.0
    
    attn = attn + rel_pos_bias.unsqueeze(0)  # [B_, num_heads, N, N]

    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = triton_softmax(attn, dim=-1)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = torch.nn.functional.linear(x, proj_weight, proj_bias)
    return x


def swin_block_forward(
    H, W,
    norm1_w, norm1_b,
    norm2_w, norm2_b,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    num_heads,
    window_size=7,
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
    x_windows = window_partition(x, window_size)  # [nW*B, ws, ws, C]
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Build qkv_bias with zeros for k
    zeros_k = torch.zeros_like(q_bias)
    qkv_bias = torch.cat([q_bias, zeros_k, v_bias])

    # Window attention
    attn_out = swin_window_attention(
        x_windows, qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        logit_scale,
        cpb_mlp_0_weight, cpb_mlp_0_bias,
        cpb_mlp_2_weight,
        relative_coords_table,
        relative_position_index,
        num_heads,
        attn_mask=attn_mask,
    )

    # Reverse window
    attn_out = attn_out.view(-1, window_size, window_size, C)
    x = window_reverse(attn_out, window_size, H, W)

    # Reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

    x = x.view(B, H * W, C)

    # Residual
    x = shortcut + x

    # MLP
    shortcut2 = x
    x = triton_layer_norm(x, norm2_w, norm2_b)
    x = torch.nn.functional.linear(x, mlp_fc1_weight, mlp_fc1_bias)
    x = triton_gelu(x)
    x = torch.nn.functional.linear(x, mlp_fc2_weight, mlp_fc2_bias)
    x = shortcut2 + x

    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    """Patch merging (downsampling)."""
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
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