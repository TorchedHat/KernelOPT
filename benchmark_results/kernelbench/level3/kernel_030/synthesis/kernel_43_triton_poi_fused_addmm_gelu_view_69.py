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
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    
    # Two-pass: first compute mean
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_row + c, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.where(mask, x, 0.0)
    mean = tl.sum(_sum, axis=0) / N
    
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_row + c, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        _var += diff * diff
    var = tl.sum(_var, axis=0) / N
    
    rstd = libdevice.rsqrt(var + eps)
    
    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_row + c, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row * stride_row + c, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N)
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        M, N, eps,
        x_2d.stride(0),
    )
    return out.reshape(orig_shape)


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
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# ============================================================
# Fused sigmoid kernel
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
def sigmoid_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
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


# ============================================================
# Fused softmax kernel (row-wise)
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
def softmax_kernel(x_ptr, out_ptr, M, N, stride_row, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Find max
    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_row + c, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.where(mask, x, float('-inf')))
    max_val = tl.max(row_max, axis=0)
    
    # Compute sum of exp
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_row + c, mask=mask, other=float('-inf')).to(tl.float32)
        _sum += tl.where(mask, libdevice.exp(x - max_val), 0.0)
    sum_val = tl.sum(_sum, axis=0)
    
    # Store normalized
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(x_ptr + row * stride_row + c, mask=mask, other=float('-inf')).to(tl.float32)
        out = libdevice.exp(x - max_val) / sum_val
        tl.store(out_ptr + row * stride_row + c, out.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    assert dim == -1 or dim == x.dim() - 1
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N).contiguous()
    out = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, out, M, N, x_2d.stride(0))
    return out.reshape(orig_shape)


# ============================================================
# Fused ReLU kernel
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
def relu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, N)
    return out


# ============================================================
# Linear (matmul + bias) using torch.mm (cuBLAS)
# ============================================================
def linear(x, weight, bias=None):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    out = x_2d @ weight.t()
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], weight.shape[0])


# ============================================================
# Window partition / reverse helpers
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


# ============================================================
# Cosine attention with relative position bias
# ============================================================
def cosine_attention(q, k, v, logit_scale, rel_pos_bias, attn_mask=None):
    # q, k, v: [B_w, num_heads, N, head_dim]
    # Normalize q and k
    q_norm = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k_norm = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Compute scale
    scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
    
    # Attention scores
    attn = (q_norm @ k_norm.transpose(-2, -1)) * scale
    
    # Add relative position bias
    attn = attn + rel_pos_bias.unsqueeze(0)
    
    # Add attention mask if provided
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(-1, nW, attn.shape[1], attn.shape[2], attn.shape[3])
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, attn.shape[2], attn.shape[3], attn.shape[4])
    
    attn = triton_softmax(attn, dim=-1)
    # dropout is identity at inference
    out = attn @ v
    return out


def compute_rel_pos_bias(relative_coords_table, relative_position_index, cpb_mlp_0_w, cpb_mlp_0_b, cpb_mlp_2_w, num_heads):
    # relative_coords_table: [1, 2*Wh-1, 2*Ww-1, 2]
    # MLP: linear -> relu -> linear
    x = relative_coords_table.view(-1, 2)  # [169, 2]
    x = linear(x, cpb_mlp_0_w, cpb_mlp_0_b)  # [169, 512]
    x = triton_relu(x)
    x = linear(x, cpb_mlp_2_w, None)  # [169, num_heads]
    x = x.view(-1, num_heads)  # [169, num_heads]
    
    # Index by relative_position_index
    idx = relative_position_index.view(-1)  # [49*49]
    rel_pos_bias = x[idx].view(49, 49, num_heads)  # [49, 49, num_heads]
    rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # [num_heads, 49, 49]
    rel_pos_bias = triton_sigmoid(rel_pos_bias) * 16.0
    return rel_pos_bias


def swin_attention_block(x, qkv_weight, q_bias, v_bias, proj_weight, proj_bias,
                          logit_scale, relative_coords_table, relative_position_index,
                          cpb_mlp_0_w, cpb_mlp_0_b, cpb_mlp_2_w,
                          num_heads, window_size, H, W, shift_size=0, attn_mask=None):
    B, L, C = x.shape
    head_dim = C // num_heads
    
    # QKV bias
    qkv_bias = torch.cat([q_bias, torch.zeros_like(v_bias), v_bias])
    
    # Reshape to spatial
    x_spatial = x.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        x_spatial = torch.roll(x_spatial, shifts=(-shift_size, -shift_size), dims=(1, 2))
    
    # Window partition
    x_windows = window_partition(x_spatial, window_size)  # [nW*B, ws, ws, C]
    x_windows = x_windows.view(-1, window_size * window_size, C)  # [nW*B, ws*ws, C]
    
    # QKV projection
    qkv = linear(x_windows, qkv_weight, qkv_bias)  # [nW*B, ws*ws, 3*C]
    qkv = qkv.reshape(-1, window_size * window_size, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, nW*B, num_heads, ws*ws, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Compute relative position bias
    rel_pos_bias = compute_rel_pos_bias(
        relative_coords_table, relative_position_index,
    )
    
    # Attention
    attn_out = cosine_attention(q, k, v, logit_scale, rel_pos_bias, attn_mask)
    
    # Reshape
    attn_out = attn_out.transpose(1, 2).reshape(-1, window_size * window_size, C)
    
    # Projection
    attn_out = linear(attn_out, proj_weight, proj_bias)
    
    # Window reverse
    attn_out = attn_out.view(-1, window_size, window_size, C)
    x_out = window_reverse(attn_out, window_size, H, W)
    
    # Reverse cyclic shift
    if shift_size > 0:
        x_out = torch.roll(x_out, shifts=(shift_size, shift_size), dims=(1, 2))
    
    x_out = x_out.view(B, H * W, C)
    return x_out


def swin_block(x, H, W,
               qkv_weight, q_bias, v_bias, proj_weight, proj_bias,
               logit_scale, relative_coords_table, relative_position_index,
               cpb_mlp_0_w, cpb_mlp_0_b, cpb_mlp_2_w,
               norm1_weight, norm1_bias,
               mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
               norm2_weight, norm2_bias,
               num_heads, window_size, shift_size=0, attn_mask=None):
    B, L, C = x.shape
    
    # Pre-norm
    x_norm = triton_layer_norm(x, norm1_weight, norm1_bias)
    
    # Attention
    attn_out = swin_attention_block(
        x_norm, qkv_weight, q_bias, v_bias, proj_weight, proj_bias,
        logit_scale, relative_coords_table, relative_position_index,
        cpb_mlp_0_w, cpb_mlp_0_b, cpb_mlp_2_w,
    )
    
    # Residual
    x = x + attn_out
    
    # MLP
    x_norm2 = triton_layer_norm(x, norm2_weight, norm2_bias)
    mlp_out = linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = triton_gelu(mlp_out)
    mlp_out = linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)
    
    x = x + mlp_out
    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    
    x_cat = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
    x_cat = x_cat.view(B, -1, 4 * C)
    
    x_out = linear(x_cat, reduction_weight, None)
    x_out = triton_layer_norm(x_out, norm_weight, norm_bias)
    return x_out, H // 2, W // 2


# ============================================================
# Main kernel_function implementing Swin Transformer V2
# ============================================================
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