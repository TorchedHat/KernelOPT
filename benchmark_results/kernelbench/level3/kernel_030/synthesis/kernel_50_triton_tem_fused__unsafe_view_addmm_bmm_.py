import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, D,
    stride_n,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    
    x_start = x_ptr + row * stride_n
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < D
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / D
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < D
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / D
    rstd = libdevice.rsqrt(var + eps)
    
    # Normalize and apply weight/bias
    out_start = out_ptr + row * stride_n
    for off in range(0, D, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < D
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_start + cols, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def layernorm_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, D,
    stride_n,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    
    x_start = x_ptr + row * stride_n
    
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < D
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / D
    
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < D
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / D
    rstd = libdevice.rsqrt(var + eps)
    
    out_start = out_ptr + row * stride_n
    for off in range(0, D, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < D
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = (x - mean) * rstd * w + b
        # GELU approximation
        c = 0.7978845608028654
        val = normed * (0.5 * (1.0 + libdevice.tanh(c * (normed + 0.044715 * normed * normed * normed))))
        tl.store(out_start + cols, val.to(tl.float32), mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N, D = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (N,)
    layernorm_kernel[grid](
        x_2d, weight, bias, out,
        N, D,
        x_2d.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


# Swin Transformer V2 attention with cosine similarity
def swin_attn(
    qkv_weight, q_bias, v_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    proj_weight, proj_bias,
    num_heads,
    pretrained_window_size=0,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads
    
    # Build QKV bias
    zeros = torch.zeros(C, device=x_windows.device, dtype=x_windows.dtype)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])
    
    # QKV projection
    qkv = torch.nn.functional.linear(x_windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: [3, B_, num_heads, N, head_dim]
    # Wait, let me re-check the reshape
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    # [3, B_, num_heads, N, head_dim]
    q, k, v = qkv.unbind(0)
    
    # Cosine attention
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    
    # logit scale
    logit_scale_val = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0/0.01, device=logit_scale.device))).exp()
    
    attn = (q @ k.transpose(-2, -1)) * logit_scale_val
    
    # Relative position bias via MLP
    rel_pos_bias_table = torch.nn.functional.linear(
        relative_coords_table.reshape(-1, 2),
    )
    rel_pos_bias_table = torch.nn.functional.relu(rel_pos_bias_table)
    rel_pos_bias_table = torch.nn.functional.linear(rel_pos_bias_table, cpb_mlp_2_weight)
    rel_pos_bias_table = rel_pos_bias_table.reshape(1, -1, num_heads)  # [1, (2Wh-1)*(2Ww-1), nH]
    
    # Index into table
    idx = relative_position_index.reshape(-1)
    rel_pos_bias = rel_pos_bias_table[:, idx, :].reshape(N, N, num_heads)
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # [nH, N, N]
    rel_pos_bias = 16 * torch.sigmoid(rel_pos_bias)
    
    attn = attn + rel_pos_bias.unsqueeze(0)
    
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)
    
    attn = torch.nn.functional.softmax(attn, dim=-1)
    
    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = torch.nn.functional.linear(x, proj_weight, proj_bias)
    return x


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


def swin_block(
    H, W,
    norm1_w, norm1_b,
    qkv_weight, q_bias, v_bias,
    logit_scale,
    cpb_mlp_0_w, cpb_mlp_0_b, cpb_mlp_2_w,
    proj_weight, proj_bias,
    norm2_w, norm2_b,
    mlp_fc1_w, mlp_fc1_b,
    mlp_fc2_w, mlp_fc2_b,
    relative_coords_table,
    relative_position_index,
    attn_mask,
    num_heads,
    window_size=7,
    shift_size=0,
):
    B, L, C = x.shape
    shortcut = x
    
    # Norm1
    x_norm = triton_layernorm(x, norm1_w, norm1_b)
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
    attn_windows = swin_attn(
        x_windows, qkv_weight, q_bias, v_bias,
        logit_scale, cpb_mlp_0_w, cpb_mlp_0_b, cpb_mlp_2_w,
        proj_weight, proj_bias,
        relative_coords_table, relative_position_index,
        attn_mask, num_heads,
    )
    
    # Reverse window
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)
    
    if shift_size > 0:
        x_out = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x_out = shifted_x
    
    x_out = x_out.view(B, H * W, C)
    x = shortcut + x_out
    
    # MLP
    x_norm2 = triton_layernorm(x, norm2_w, norm2_b)
    # FC1 + GELU
    x_mlp = torch.nn.functional.linear(x_norm2, mlp_fc1_w, mlp_fc1_b)
    x_mlp = torch.nn.functional.gelu(x_mlp)
    # FC2
    x_mlp = torch.nn.functional.linear(x_mlp, mlp_fc2_w, mlp_fc2_b)
    x = x + x_mlp
    
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
    
    x = torch.nn.functional.linear(x, reduction_weight)
    x = triton_layernorm(x, norm_w, norm_b)
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
    p_layers_2_blocks_5_mlp_fc1_bias,
    p_layers_2_blocks_5_mlp_fc2_weight,
    p_layers_2_blocks_5_mlp_fc2_bias,
    p_layers_2_blocks_5_norm2_weight,
    p_layers_2_blocks_5_norm2_bias,
    p_layers_2_downsample_reduction_weight,
    p_layers_2_downsample_norm_weight,
    p_layers_2_downsample_norm_bias,
    p_layers_3_blocks_0_attn_q_bias,
    p_layers_3_blocks_0_attn_v_bias,
    p_layers_3_blocks_0_attn_qkv_weight,
    p_layers_3_blocks_0_attn_logit_scale,
    p_layers_3_blocks_0_attn_cpb_mlp_0_weight,
    p_layers_3_blocks_0_attn_cpb_mlp_0_bias,
    p_layers_3_blocks_0_attn_cpb_mlp_2_weight,
    p_layers_3_blocks_0_attn_proj_weight,
    p_layers_3_blocks_0_attn_proj_bias,
    p_layers_3_blocks_0_norm1_weight,
    p_layers_3_blocks_0_norm1_bias,
    p_layers_3_blocks_0_mlp_fc1_weight,
    p_layers_3_blocks_0_mlp_fc1_bias,
    p_layers_3_blocks_0_mlp_fc2_weight,
    p_layers_3_blocks_0_mlp_fc2_bias,
    p_layers_3_blocks_0_norm2_weight,
    p_layers_3_blocks_0_norm2_bias,
    p_layers_3_blocks_1_attn_q_bias,