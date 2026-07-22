import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    stride_m,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = libdevice.rsqrt(var + eps)
    
    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row * stride_m + cols, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    M, N,
    stride_m,
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
        x = tl.load(in_ptr + row * stride_m + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(in_ptr + row * stride_m + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    # Store normalized
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(in_ptr + row * stride_m + cols, mask=mask, other=float('-inf')).to(tl.float32)
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_ptr + row * stride_m + cols, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
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
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        M, N,
        x_2d.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


def triton_softmax(x):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](
        out, x_2d,
        M, N,
        x_2d.stride(0),
    )
    return out.reshape(orig_shape)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
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


def compute_relative_position_bias(relative_coords_table, relative_position_index, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight, num_heads, window_size=7):
    # MLP on relative coords table
    rct = relative_coords_table  # [1, 13, 13, 2]
    # fc1
    h = rct.reshape(-1, 2) @ cpb_mlp_0_weight.T + cpb_mlp_0_bias
    h = torch.relu(h)
    # fc2 (no bias)
    h = h @ cpb_mlp_2_weight.T  # [169, num_heads]
    h = h.view(13*13, num_heads)
    
    # Index
    idx = relative_position_index.view(-1)  # [49*49]
    rpb = h[idx].view(window_size*window_size, window_size*window_size, num_heads)
    rpb = rpb.permute(2, 0, 1).contiguous()  # [num_heads, 49, 49]
    rpb = torch.sigmoid(rpb) * 16.0
    return rpb


def swin_attention(x, qkv_weight, q_bias, v_bias, proj_weight, proj_bias,
                   logit_scale, relative_coords_table, relative_position_index,
                   cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
                   num_heads, window_size, attn_mask=None):
    B_, N, C = x.shape
    head_dim = C // num_heads
    
    # Build qkv bias
    k_bias = torch.zeros_like(q_bias)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias])
    
    # QKV projection
    qkv = x.reshape(B_ * N, C) @ qkv_weight.T + qkv_bias
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: [3, B_, num_heads, N, head_dim] -- wait, let me check
    # Actually permute(2,0,1,3,4) on [B_, N, 3, num_heads, head_dim] gives [3, B_, N, num_heads, head_dim]
    # But we need [3, B_, num_heads, N, head_dim]
    qkv = x.reshape(B_ * N, C) @ qkv_weight.T + qkv_bias
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
    
    q, k, v = qkv[0], qkv[1], qkv[2]  # each [B_, num_heads, N, head_dim]
    
    # Normalize q and k
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)
    
    # Logit scale
    scale = torch.clamp(logit_scale, max=torch.log(torch.tensor(1.0/0.01, device=logit_scale.device))).exp()
    
    # Attention
    attn = q_norm @ k_norm.transpose(-2, -1) * scale  # [B_, num_heads, N, N]
    
    # Relative position bias
    rpb = compute_relative_position_bias(
        relative_coords_table, relative_position_index,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    )
    attn = attn + rpb.unsqueeze(0)
    
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(B_, num_heads, N, N)
    
    attn = triton_softmax(attn)
    
    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = x @ proj_weight.T + proj_bias
    return x


def swin_block(x, norm1_w, norm1_b, norm2_w, norm2_b,
               qkv_weight, q_bias, v_bias, proj_weight, proj_bias,
               logit_scale, relative_coords_table, relative_position_index,
               cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
               mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
               num_heads, window_size, H, W, shift_size=0, attn_mask=None):
    B, L, C = x.shape
    
    shortcut = x
    x_ln = triton_layer_norm(x, norm1_w, norm1_b)
    x_2d = x_ln.view(B, H, W, C)
    
    if shift_size > 0:
        x_2d = torch.roll(x_2d, shifts=(-shift_size, -shift_size), dims=(1, 2))
    
    # Window partition
    x_windows = window_partition(x_2d, window_size)  # [nW*B, ws, ws, C]
    x_windows = x_windows.view(-1, window_size * window_size, C)
    
    # Attention
    attn_out = swin_attention(
        x_windows, qkv_weight, q_bias, v_bias, proj_weight, proj_bias,
        logit_scale, relative_coords_table, relative_position_index,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    )
    
    # Window reverse
    attn_out = attn_out.view(-1, window_size, window_size, C)
    x_2d = window_reverse(attn_out, window_size, H, W)
    
    if shift_size > 0:
        x_2d = torch.roll(x_2d, shifts=(shift_size, shift_size), dims=(1, 2))
    
    x = x_2d.view(B, H * W, C)
    x = shortcut + x
    
    # MLP
    shortcut2 = x
    x = triton_layer_norm(x, norm2_w, norm2_b)
    x = x.reshape(-1, C) @ mlp_fc1_weight.T + mlp_fc1_bias
    x = triton_gelu(x)
    x = x.reshape(-1, mlp_fc1_weight.shape[0]) @ mlp_fc2_weight.T + mlp_fc2_bias
    x = x.reshape(B, L, C)
    x = shortcut2 + x
    
    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
    x = x.view(B, -1, 4 * C)
    
    x = x.reshape(-1, 4*C) @ reduction_weight.T
    x = x.reshape(B, -1, reduction_weight.shape[0])
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
    p_layers_2_blocks_5_mlp_fc1_bias,