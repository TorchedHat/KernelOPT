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
        triton.Config({'BLOCK_SIZE': 16, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 16, 'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8, 'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N_ROWS', 'N_COLS'],
)
@triton.jit
def fused_layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N_ROWS, N_COLS,
    stride_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_SIZE
    row_offs = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offs < N_ROWS

    # Compute mean and variance
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_off in range(0, N_COLS, BLOCK_N):
        col_offs = col_off + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N_COLS
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + row_offs[:, None] * stride_row + col_offs[None, :],
                     mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=1)

    mean = mean / N_COLS

    for col_off in range(0, N_COLS, BLOCK_N):
        col_offs = col_off + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N_COLS
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + row_offs[:, None] * stride_row + col_offs[None, :],
                     mask=mask, other=0.0).to(tl.float32)
        diff = x - mean[:, None]
        var += tl.sum(diff * diff, axis=1)

    var = var / N_COLS
    rstd = 1.0 / tl.sqrt(var + eps)

    for col_off in range(0, N_COLS, BLOCK_N):
        col_offs = col_off + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N_COLS
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + row_offs[:, None] * stride_row + col_offs[None, :],
                     mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offs, mask=col_mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
        x_norm = (x - mean[:, None]) * rstd[:, None]
        out = x_norm * w[None, :] + b[None, :]
        tl.store(out_ptr + row_offs[:, None] * stride_row + col_offs[None, :],
                 out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N_ROWS, N_COLS = x_2d.shape
    out = torch.empty_like(x_2d)
    
    grid = lambda meta: (triton.cdiv(N_ROWS, meta['BLOCK_SIZE']),)
    fused_layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        N_ROWS, N_COLS,
        x_2d.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
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
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_val = libdevice.tanh(inner)
    out = x * 0.5 * (1.0 + tanh_val)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, N)
    return out


# Softmax kernel - one row per program
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N_COLS'],
)
@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    N_ROWS, N_COLS,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= N_ROWS:
        return
    
    row_start = x_ptr + row_idx * stride_row
    out_start = out_ptr + row_idx * stride_row
    
    # Find max
    row_max = tl.full([], float('-inf'), dtype=tl.float32)
    for col_off in range(0, N_COLS, BLOCK_N):
        col_offs = col_off + tl.arange(0, BLOCK_N)
        mask = col_offs < N_COLS
        x = tl.load(row_start + col_offs, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros([], dtype=tl.float32)
    for col_off in range(0, N_COLS, BLOCK_N):
        col_offs = col_off + tl.arange(0, BLOCK_N)
        mask = col_offs < N_COLS
        x = tl.load(row_start + col_offs, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    # Normalize
    for col_off in range(0, N_COLS, BLOCK_N):
        col_offs = col_off + tl.arange(0, BLOCK_N)
        mask = col_offs < N_COLS
        x = tl.load(row_start + col_offs, mask=mask, other=float('-inf')).to(tl.float32)
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_start + col_offs, out.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Flatten to 2D with last dim as reduction
    orig_shape = x.shape
    if dim < 0:
        dim = len(orig_shape) + dim
    
    # Move reduction dim to last
    if dim != len(orig_shape) - 1:
        x = x.transpose(dim, -1).contiguous()
    
    x_2d = x.reshape(-1, x.shape[-1])
    N_ROWS, N_COLS = x_2d.shape
    out = torch.empty_like(x_2d)
    
    grid = (N_ROWS,)
    softmax_kernel[grid](
        x_2d, out,
        N_ROWS, N_COLS,
        x_2d.stride(0),
    )
    
    out = out.reshape(x.shape)
    if dim != len(orig_shape) - 1:
        out = out.transpose(dim, -1).contiguous()
    return out.reshape(orig_shape)


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
    qkv_weight, q_bias, v_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    proj_weight, proj_bias,
    relative_coords_table,
    relative_position_index,
    num_heads,
    attn_mask=None,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads
    
    # Build qkv bias
    zeros = torch.zeros(C, device=x_windows.device, dtype=x_windows.dtype)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])
    
    # QKV projection
    qkv = torch.nn.functional.linear(x_windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # Wait, reshape should be [B_, N, 3, num_heads, head_dim]
    # permute to [3, B_, num_heads, N, head_dim]
    qkv = x_windows @ qkv_weight.T + qkv_bias
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    # Cosine attention
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)
    
    # Logit scale
    scale = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0/0.01, device=logit_scale.device))).exp()
    
    attn = (q_norm @ k_norm.transpose(-2, -1)) * scale
    
    # Relative position bias via MLP
    rel_pos_bias_table = torch.nn.functional.linear(relative_coords_table, cpb_mlp_0_weight, cpb_mlp_0_bias)
    rel_pos_bias_table = torch.nn.functional.relu(rel_pos_bias_table)
    rel_pos_bias_table = torch.nn.functional.linear(rel_pos_bias_table, cpb_mlp_2_weight)
    rel_pos_bias_table = rel_pos_bias_table.view(-1, num_heads)
    
    idx = relative_position_index.view(-1)
    rel_pos_bias = rel_pos_bias_table[idx].view(N, N, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
    rel_pos_bias = triton_sigmoid(rel_pos_bias) * 16
    
    attn = attn + rel_pos_bias.unsqueeze(0)
    
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)
    
    attn = triton_softmax(attn, dim=-1)
    
    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = torch.nn.functional.linear(x, proj_weight, proj_bias)
    return x


def swin_block(
    norm1_weight, norm1_bias,
    q_bias, v_bias, qkv_weight,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    proj_weight, proj_bias,
    relative_coords_table, relative_position_index,
    norm2_weight, norm2_bias,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    num_heads, window_size, shift_size,
    H, W,
    attn_mask=None,
):
    B, L, C = x.shape
    shortcut = x
    
    x = triton_layer_norm(x, norm1_weight, norm1_bias)
    x = x.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    
    # Window partition
    x_windows = window_partition(x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)
    
    # Attention
    attn_out = swin_attention(
        x_windows, qkv_weight, q_bias, v_bias,
        logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        proj_weight, proj_bias,
        relative_coords_table, relative_position_index,
    )
    
    # Reverse window
    attn_out = attn_out.view(-1, window_size, window_size, C)
    x = window_reverse(attn_out, window_size, H, W)
    
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
    
    x = x.view(B, H * W, C)
    x = shortcut + x
    
    # MLP
    x = x + mlp_forward(x, norm2_weight, norm2_bias, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias)
    
    return x


def mlp_forward(x, norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    x_norm = triton_layer_norm(x, norm_weight, norm_bias)
    x_fc1 = torch.nn.functional.linear(x_norm, fc1_weight, fc1_bias)
    x_act = triton_gelu(x_fc1)
    x_fc2 = torch.nn.functional.linear(x_act, fc2_weight, fc2_bias)
    return x_fc2


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