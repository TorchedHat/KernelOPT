import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 16}),
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
def fused_layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    stride_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_row
    out_ptr = out_ptr + row * stride_row

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    fused_layer_norm_kernel[grid](
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
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
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
    y = x * cdf
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# Fused softmax kernel for attention
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
    x_ptr = x_ptr + row * stride_row
    out_ptr = out_ptr + row * stride_row

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


def triton_softmax(x, dim=-1):
    # x: arbitrary shape, softmax over last dim
    orig_shape = x.shape
    if dim != -1 and dim != len(orig_shape) - 1:
        # permute to make dim last
        x = x.transpose(dim, -1).contiguous()
        result = triton_softmax(x, dim=-1)
        return result.transpose(dim, -1).contiguous()
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, out, M, N, x_2d.stride(0))
    return out.reshape(orig_shape)


# Fused sigmoid kernel
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
def sigmoid_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, N)
    return out


# Fused linear + bias kernel (for small matrices where we want to fuse bias add)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def triton_linear(x, weight, bias=None):
    """x: (..., K), weight: (N, K), bias: (N,) -> (..., N)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    
    # weight is (N, K), we need x @ weight.T = (M, K) @ (K, N)
    weight_t = weight.t().contiguous()
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    matmul_bias_kernel[grid](
        x_2d, weight_t, bias if bias is not None else x_2d,
        out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )
    
    return out.reshape(orig_shape[:-1] + (N,))


# ============================================================
# Swin Transformer V2 implementation using Triton kernels
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
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    num_heads,
    window_size=7,
):
    B_, N, C = x.shape
    head_dim = C // num_heads

    # Build qkv bias
    zeros = torch.zeros(C, device=x.device, dtype=x.dtype)
    qkv_bias = torch.cat([q_bias, zeros, v_bias])

    # QKV projection
    qkv = triton_linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    # qkv: (3, B_, num_heads, N, head_dim) -- wait, let me check
    # Actually reshape is (B_, N, 3, num_heads, head_dim), permute(2,0,3,1,4)
    qkv = x.reshape(B_, N, -1)  # placeholder - redo
    
    # Redo properly
    qkv_out = triton_linear(x, qkv_weight, qkv_bias)
    qkv_out = qkv_out.reshape(B_, N, 3, num_heads, head_dim)
    qkv_out = qkv_out.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
    
    q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]  # each: (B_, num_heads, N, head_dim)

    # Normalize q and k
    q_norm = torch.nn.functional.normalize(q, dim=-1)
    k_norm = torch.nn.functional.normalize(k, dim=-1)

    # Logit scale
    logit_scale_val = torch.clamp(logit_scale, max=math.log(100.0)).exp()

    # Attention scores
    attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * logit_scale_val

    # Relative position bias via CPB MLP
    # cpb_mlp: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
    cpb_out = triton_linear(relative_coords_table.view(-1, 2), cpb_mlp_0_weight, cpb_mlp_0_bias)
    cpb_out = torch.relu(cpb_out)
    cpb_out = triton_linear(cpb_out, cpb_mlp_2_weight, None)
    
    relative_position_bias_table = cpb_out.view(-1, num_heads)
    
    # Index into table
    idx = relative_position_index.view(-1)
    relative_position_bias = relative_position_bias_table[idx].view(
    )
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    relative_position_bias = triton_sigmoid(relative_position_bias) * 16.0
    
    attn = attn + relative_position_bias.unsqueeze(0)

    # Apply attention mask if provided
    if attn_mask is not None:
        nW = attn_mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)

    attn = triton_softmax(attn, dim=-1)

    # Attention output
    x_out = torch.matmul(attn, v)  # (B_, num_heads, N, head_dim)
    x_out = x_out.transpose(1, 2).reshape(B_, N, C)

    # Output projection
    x_out = triton_linear(x_out, proj_weight, proj_bias)

    return x_out


def swin_block(
    norm1_weight, norm1_bias,
    qkv_weight, q_bias, v_bias,
    proj_weight, proj_bias,
    logit_scale,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    relative_coords_table,
    relative_position_index,
    attn_mask,
    norm2_weight, norm2_bias,
    mlp_fc1_weight, mlp_fc1_bias,
    mlp_fc2_weight, mlp_fc2_bias,
    num_heads,
    window_size=7,
    shift_size=0,
    H=56, W=56,
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
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight,
        relative_coords_table, relative_position_index,
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

    # Residual + Norm2 + MLP
    x = shortcut + x_out
    x_norm2 = triton_layer_norm(x, norm2_weight, norm2_bias)

    # MLP
    mlp_out = triton_linear(x_norm2, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = triton_gelu(mlp_out)
    mlp_out = triton_linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)

    x = x + mlp_out
    return x


def patch_merging(x, H, W, reduction_weight, norm_weight, norm_bias):
    """Patch merging / downsampling"""
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
    x_cat = x_cat.view(B, -1, 4 * C)

    # Linear reduction (no bias)
    x_out = triton_linear(x_cat, reduction_weight, None)
    x_out = triton_layer_norm(x_out, norm_weight, norm_bias)
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