import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 256}, num_warps=8),
    ],
    key=['N', 'C'],
)
@triton.jit
def fused_layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return
    
    row_start = row_idx * C
    
    # Compute mean and variance
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < C
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / C
    
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < C
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / C
    
    rstd = libdevice.rsqrt(var + eps)
    
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < C
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask)


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch.nn.functional as F

# ============================================================
# Fused LayerNorm Triton Kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32}, num_warps=2, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def layernorm_fwd_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C,
    eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return

    row_start = row_idx * C

    # Pass 1: compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < C
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / C

    # Pass 2: compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < C
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / C
    rstd = libdevice.rsqrt(var + eps)

    # Pass 3: normalize and store
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < C
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N, C = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (N,)
    layernorm_fwd_kernel[grid](
        x_2d, weight, bias, out,
        N, C, eps,
    )
    return out.reshape(orig_shape)


# ============================================================
# Fused GELU Triton Kernel
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
def gelu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# ============================================================
# Fused Softmax Triton Kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
    ],
    key=['N_COLS'],
)
@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    N_ROWS, N_COLS,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= N_ROWS:
        return

    row_start = row_idx * N_COLS

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N_COLS, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N_COLS, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Normalize and store
    for off in range(0, N_COLS, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Flatten to 2D with last dim as reduction
    orig_shape = x.shape
    if dim < 0:
        dim = len(orig_shape) + dim
    # Move reduction dim to last
    if dim != len(orig_shape) - 1:
        x = x.transpose(dim, -1).contiguous()
        transposed = True
    else:
        x = x.contiguous()
        transposed = False

    x_2d = x.reshape(-1, x.shape[-1])
    N_ROWS, N_COLS = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (N_ROWS,)
    softmax_kernel[grid](x_2d, out, N_ROWS, N_COLS)
    out = out.reshape(x.shape)
    if transposed:
        out = out.transpose(dim, -1).contiguous()
    return out.reshape(orig_shape)


# ============================================================
# Swin Transformer V2 Implementation
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
    q_bias, v_bias,
    qkv_weight,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    proj_weight, proj_bias,
    num_heads,
    window_size,
):
    B_, N, C = x.shape
    head_dim = C // num_heads

    # Build qkv bias
    k_bias = torch.zeros_like(q_bias)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias])

    # QKV projection
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: (3, B_, num_heads, N, head_dim) -> wait, let me check
    # Actually reshape to (B_, N, 3, num_heads, head_dim) then permute(2, 0, 3, 1, 4)
    qkv = x.reshape(B_, N, 1)  # placeholder - redo
    
    # Redo properly
    qkv_out = F.linear(x, qkv_weight, qkv_bias)
    qkv_out = qkv_out.reshape(B_, N, 3, num_heads, head_dim)
    qkv_out = qkv_out.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
    q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]

    # Cosine attention
    q_norm = F.normalize(q, dim=-1)
    k_norm = F.normalize(k, dim=-1)

    # Logit scale
    logit_scale_clamped = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=logit_scale.device))).exp()
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * logit_scale_clamped

    # Relative position bias via MLP
    # cpb_mlp: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
    rel_pos_bias_table = F.linear(relative_coords_table, cpb_mlp_0_weight, cpb_mlp_0_bias)
    rel_pos_bias_table = F.relu(rel_pos_bias_table)
    rel_pos_bias_table = F.linear(rel_pos_bias_table, cpb_mlp_2_weight)
    rel_pos_bias_table = rel_pos_bias_table.view(-1, num_heads)

    # Index into table
    rel_pos_index = relative_position_index.view(-1)
    rel_pos_bias = rel_pos_bias_table[rel_pos_index].view(
    )
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16
    attn = attn + rel_pos_bias.unsqueeze(0)

    # Apply mask if shifted window
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = triton_softmax(attn, dim=-1)

    x_out = torch.matmul(attn, v)
    x_out = x_out.transpose(1, 2).reshape(B_, N, C)
    x_out = F.linear(x_out, proj_weight, proj_bias)
    return x_out


def swin_block(
    # attention params
    q_bias, v_bias, qkv_weight, logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    proj_weight, proj_bias,
    norm1_weight, norm1_bias,
    # mlp params
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    norm2_weight, norm2_bias,
    # attention config
    relative_coords_table, relative_position_index,
    attn_mask,
    num_heads, window_size, shift_size,
    H, W,
):
    B, L, C = x.shape
    shortcut = x

    # Norm1
    x_norm = triton_layernorm(x, norm1_weight, norm1_bias)
    x_norm = x_norm.view(B, H, W, C)

    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x_norm, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x_norm

    # Window partition
    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    # Window attention
    attn_windows = swin_window_attention(
        x_windows, q_bias, v_bias, qkv_weight, logit_scale,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        proj_weight, proj_bias,
        relative_coords_table, relative_position_index,
        attn_mask, num_heads, window_size,
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
    shortcut2 = x
    x_norm2 = triton_layernorm(x, norm2_weight, norm2_bias)
    x_mlp = F.linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    x_mlp = triton_gelu(x_mlp)
    x_mlp = F.linear(x_mlp, mlp_fc2_weight, mlp_fc2_bias)
    x = shortcut2 + x_mlp

    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
    x_cat = x_cat.view(B, -1, 4 * C)

    x_out = F.linear(x_cat, reduction_weight)
    x_out = triton_layernorm(x_out, norm_weight, norm_bias)
    return x_out


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