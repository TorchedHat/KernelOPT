import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math

# ============================================================
# Triton kernels for fused operations
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
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_m
    out_ptr = out_ptr + row * stride_m

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
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


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
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


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
def softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_m
    out_ptr = out_ptr + row * stride_m

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


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        M, N, eps,
        x_2d.stride(0),
    )
    return out.reshape(orig_shape)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


def triton_softmax(x, dim=-1):
    # Reshape so last dim is the reduction dim
    orig_shape = x.shape
    if dim != -1 and dim != len(orig_shape) - 1:
        x = x.transpose(dim, -1).contiguous()
        transposed = True
    else:
        x = x.contiguous()
        transposed = False
    
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](
        x_2d, out,
        M, N,
        x_2d.stride(0),
    )
    out = out.reshape(x.shape)
    if transposed:
        out = out.transpose(dim, -1).contiguous()
    return out


# ============================================================
# SwinV2 implementation using Triton kernels for fusable ops
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


def window_attention_forward(
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    relative_coords_table, relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    num_heads,
    window_size=7,
):
    B_, N, C = x.shape
    head_dim = C // num_heads

    # Build qkv bias
    qkv_bias = torch.cat([q_bias, torch.zeros_like(v_bias), v_bias])
    
    # QKV projection
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: [3, B_, num_heads, N, head_dim] -> wait, let me check
    # reshape: [B_, N, 3, num_heads, head_dim]
    # permute(2,0,1,3,4) -> [3, B_, num_heads, N, head_dim]
    # Hmm, but the ATen trace shows [3, B_, num_heads, N, head_dim]
    # Actually from the trace: reshape -> [640, 49, 3, 3, 32], permute -> [3, 640, 3, 49, 32]
    # So it's [3, B_, num_heads, N, head_dim]
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    # Wait: permute(2,0,1,3,4) on [B_, N, 3, num_heads, head_dim]
    # -> dim0=3(=3), dim1=B_, dim2=N, dim3=num_heads, dim4=head_dim
    # But trace shows permute.default -> shape=[3, 640, 3, 49, 32]
    # So [3, B_, num_heads, N, head_dim] - that's permute(2,0,3,1,4)? No...
    # [B_, N, 3, num_heads, head_dim] with permute(2,0,3,1,4):
    # result[i,j,k,l,m] = input[j,l,i,k,m]
    # shape: [3, B_, num_heads, N, head_dim] ✓
    
    q, k, v = qkv[0], qkv[1], qkv[2]  # each [B_, num_heads, N, head_dim]

    # Normalize q and k
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)

    # Logit scale
    logit_scale_clamped = torch.clamp(logit_scale, max=math.log(100.0)).exp()
    
    # Attention: q @ k^T * scale
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * logit_scale_clamped

    # Relative position bias via CPB MLP
    # relative_coords_table: [1, 2*(ws-1), 2*(ws-1), 2]
    # cpb_mlp: linear -> relu -> linear
    rel_pos_bias_table = F.linear(relative_coords_table, cpb_mlp_0_weight, cpb_mlp_0_bias)
    rel_pos_bias_table = F.relu(rel_pos_bias_table)
    rel_pos_bias_table = F.linear(rel_pos_bias_table, cpb_mlp_2_weight)
    rel_pos_bias_table = rel_pos_bias_table.view(-1, num_heads)  # [169, num_heads]
    
    # Index into table
    rel_pos_index = relative_position_index.view(-1)  # [N*N]
    rel_pos_bias = rel_pos_bias_table[rel_pos_index]  # [N*N, num_heads]
    rel_pos_bias = rel_pos_bias.view(N, N, num_heads).permute(2, 0, 1).contiguous()  # [num_heads, N, N]
    rel_pos_bias = torch.sigmoid(rel_pos_bias) * 16.0
    rel_pos_bias = rel_pos_bias.unsqueeze(0)  # [1, num_heads, N, N]
    
    attn = attn + rel_pos_bias

    # Apply attention mask if provided
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    # Softmax
    attn = triton_softmax(attn, dim=-1)

    # Attention @ V
    x = torch.matmul(attn, v)  # [B_, num_heads, N, head_dim]
    x = x.transpose(1, 2).reshape(B_, N, C)
    
    # Projection
    x = F.linear(x, proj_weight, proj_bias)
    return x


def swin_block_forward(
    norm1_weight, norm1_bias,
    norm2_weight, norm2_bias,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    relative_coords_table, relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    attn_mask,
    H, W,
    num_heads,
    window_size=7,
    shift_size=0,
):
    B, L, C = x.shape
    shortcut = x
    
    # Norm1
    x = triton_layer_norm(x, norm1_weight, norm1_bias)
    x = x.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x
    
    # Window partition
    x_windows = window_partition(shifted_x, window_size)  # [nW*B, ws, ws, C]
    x_windows = x_windows.view(-1, window_size * window_size, C)
    
    # Window attention
    attn_windows = window_attention_forward(
        x_windows, qkv_weight, q_bias, v_bias,
        proj_weight, proj_bias,
        logit_scale,
        relative_coords_table, relative_position_index,
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
    )
    
    # Reverse window
    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)
    
    # Reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x
    
    x = x.view(B, H * W, C)
    x = shortcut + x
    
    # MLP
    shortcut2 = x
    x = triton_layer_norm(x, norm2_weight, norm2_bias)
    x = F.linear(x, mlp_fc1_weight, mlp_fc1_bias)
    x = triton_gelu(x)
    x = F.linear(x, mlp_fc2_weight, mlp_fc2_bias)
    x = shortcut2 + x
    
    return x


def patch_merging_forward(x, H, W, reduction_weight, norm_weight, norm_bias):
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
    x = x.view(B, -1, 4 * C)
    
    x = F.linear(x, reduction_weight)
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