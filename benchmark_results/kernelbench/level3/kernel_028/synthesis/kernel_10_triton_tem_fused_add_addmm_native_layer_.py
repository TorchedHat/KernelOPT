import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# I'll implement this as a series of Triton kernels that together implement the full ViT forward pass.
# The key insight is to fuse operations where possible.


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
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X_ptr + row * N
    Y_ptr = Y_ptr + row * N
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)
    
    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_ptr + cols, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    layer_norm_kernel[(M,)](x_2d, weight, bias, y, N, eps)
    return y.reshape(orig_shape)


# ============================================================
# GEMM kernel (tiled matrix multiplication with bias)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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
    
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


def triton_linear(x, weight, bias=None):
    """x: (..., K), weight: (N, K), bias: (N,) -> (..., N)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    
    # weight is (N, K), we want C = x @ weight.T
    # So A=x (M,K), B=weight.T (K,N)
    C = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    gemm_bias_kernel[grid](
        C,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),  # B is weight transposed: stride_bk=weight.stride(1), stride_bn=weight.stride(0)
        C.stride(0), C.stride(1),
        HAS_BIAS=(bias is not None),
    )
    
    return C.reshape(orig_shape[:-1] + (N,))


# ============================================================
# Fused patch embedding: unfold + reshape + linear
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def patch_embed_kernel(
    img_ptr, W_ptr, bias_ptr, out_ptr,
    B, C, H, W_img,
    patch_size,
    num_patches_h, num_patches_w,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    patches: (B * num_patches_h * num_patches_w, C * patch_size * patch_size) = (M, K)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_patches = num_patches_h * num_patches_w
    
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Decode offs_m -> (batch, patch_h, patch_w)
        batch_idx = offs_m // num_patches
        patch_idx = offs_m % num_patches
        ph = patch_idx // num_patches_w
        pw = patch_idx % num_patches_w
        
        # Decode offs_k -> (c, kh, kw)
        c_idx = offs_k // (patch_size * patch_size)
        rem = offs_k % (patch_size * patch_size)
        kh = rem // patch_size
        kw = rem % patch_size
        
        # Compute image indices
        ih = ph[:, None] * patch_size + kh[None, :]
        iw = pw[:, None] * patch_size + kw[None, :]
        
        img_idx = (batch_idx[:, None] * C * H * W_img + 
                   c_idx[None, :] * H * W_img + 
        
        m_mask = offs_m < M
        a = tl.load(img_ptr + img_idx, 
                    mask=(m_mask[:, None]) & k_mask[None, :] & 
                         (batch_idx[:, None] < B) & (c_idx[None, :] < C) &
                         (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W_img),
                    other=0.0)
        
        b = tl.load(W_ptr + offs_n[None, :] * K + offs_k[:, None],
                    mask=(offs_n[None, :] < N) & k_mask[:, None],
                    other=0.0)
        
        acc = tl.dot(a, b, acc=acc)
    
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# ============================================================
# Fused attention: Q,K,V projection + scaled dot product attention
# ============================================================

# We'll use PyTorch's scaled_dot_product_attention for the attention computation
# but implement the surrounding operations in Triton

# ============================================================
# Fused ReLU + Linear (FFN)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    # ReLU
    acc = tl.maximum(acc, 0.0)
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


def triton_linear_relu(x, weight, bias):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    C = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_relu_kernel[grid](
        x_2d, weight, bias, C,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),
        C.stride(0), C.stride(1),
    )
    return C.reshape(orig_shape[:-1] + (N,))


# ============================================================
# GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# ============================================================
# Fused add + layer norm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def add_layer_norm_kernel(
    X_ptr, Residual_ptr, W_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X_ptr + row * N
    Residual_ptr = Residual_ptr + row * N
    Y_ptr = Y_ptr + row * N
    
    # Compute mean of (x + residual)
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(tl.where(mask, x + r, 0.0), axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + r - mean, 0.0)
        var += tl.sum(val * val, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)
    
    # Normalize and store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x + r - mean) * rstd * w + b
        tl.store(Y_ptr + cols, y.to(tl.float32), mask=mask)


def triton_add_layer_norm(x, residual, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    r_2d = residual.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    add_layer_norm_kernel[(M,)](x_2d, r_2d, weight, bias, y, N, eps)
    return y.reshape(orig_shape)


# ============================================================
# Multi-head self-attention
# ============================================================
def triton_mha(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
    """
    """
    B, S, D = x.shape
    
    # QKV projection: (B*S, 3*D)
    qkv = triton_linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3*D)
    
    # Split into Q, K, V
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    q = qkv[:, :, 0].transpose(1, 2)  # (B, num_heads, S, head_dim)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    
    # Scaled dot product attention
    scale = head_dim ** -0.5
    attn = torch.bmm(q.reshape(B * num_heads, S, head_dim), 
                     k.reshape(B * num_heads, S, head_dim).transpose(1, 2)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.bmm(attn, v.reshape(B * num_heads, S, head_dim))
    out = out.reshape(B, num_heads, S, head_dim).transpose(1, 2).reshape(B, S, D)
    
    # Output projection
    out = triton_linear(out, out_proj_weight, out_proj_bias)
    return out


# ============================================================
# Transformer encoder layer
# ============================================================
def transformer_layer(x, 
                      in_proj_weight, in_proj_bias,
                      out_proj_weight, out_proj_bias,
                      norm1_weight, norm1_bias,
                      linear1_weight, linear1_bias,
                      linear2_weight, linear2_bias,
                      norm2_weight, norm2_bias,
                      num_heads=8, head_dim=64):
    # Self-attention with pre-norm (PyTorch TransformerEncoderLayer uses post-norm by default)
    # Actually PyTorch's default is post-norm
    
    # Self-attention
    attn_out = triton_mha(x, in_proj_weight, in_proj_bias, 
                          out_proj_weight, out_proj_bias,
    
    # Residual + LayerNorm
    x = triton_add_layer_norm(attn_out, x, norm1_weight, norm1_bias)
    
    # FFN: linear1 + relu + linear2
    ffn_out = triton_linear_relu(x, linear1_weight, linear1_bias)
    ffn_out = triton_linear(ffn_out, linear2_weight, linear2_bias)
    
    # Residual + LayerNorm
    x = triton_add_layer_norm(ffn_out, x, norm2_weight, norm2_bias)
    
    return x


# ============================================================
# Main kernel function
# ============================================================
def kernel_function(
    p_patch_to_embedding_weight,
    p_patch_to_embedding_bias,
    p_cls_token,
    p_pos_embedding,
    p_transformer_layers_0_self_attn_in_proj_bias,
    p_transformer_layers_0_self_attn_in_proj_weight,
    p_transformer_layers_0_self_attn_out_proj_weight,
    p_transformer_layers_0_self_attn_out_proj_bias,
    p_transformer_layers_0_norm1_weight,
    p_transformer_layers_0_norm1_bias,
    p_transformer_layers_0_linear1_weight,
    p_transformer_layers_0_linear1_bias,
    p_transformer_layers_0_linear2_weight,
    p_transformer_layers_0_linear2_bias,
    p_transformer_layers_0_norm2_weight,
    p_transformer_layers_0_norm2_bias,
    p_transformer_layers_1_self_attn_in_proj_bias,
    p_transformer_layers_1_self_attn_in_proj_weight,
    p_transformer_layers_1_self_attn_out_proj_weight,
    p_transformer_layers_1_self_attn_out_proj_bias,
    p_transformer_layers_1_norm1_weight,
    p_transformer_layers_1_norm1_bias,
    p_transformer_layers_1_linear1_weight,
    p_transformer_layers_1_linear1_bias,
    p_transformer_layers_1_linear2_weight,
    p_transformer_layers_1_linear2_bias,
    p_transformer_layers_1_norm2_weight,
    p_transformer_layers_1_norm2_bias,
    p_transformer_layers_2_self_attn_in_proj_bias,
    p_transformer_layers_2_self_attn_in_proj_weight,
    p_transformer_layers_2_self_attn_out_proj_weight,
    p_transformer_layers_2_self_attn_out_proj_bias,
    p_transformer_layers_2_norm1_weight,
    p_transformer_layers_2_norm1_bias,
    p_transformer_layers_2_linear1_weight,
    p_transformer_layers_2_linear1_bias,
    p_transformer_layers_2_linear2_weight,
    p_transformer_layers_2_linear2_bias,
    p_transformer_layers_2_norm2_weight,
    p_transformer_layers_2_norm2_bias,
    p_transformer_layers_3_self_attn_in_proj_bias,
    p_transformer_layers_3_self_attn_in_proj_weight,
    p_transformer_layers_3_self_attn_out_proj_weight,
    p_transformer_layers_3_self_attn_out_proj_bias,
    p_transformer_layers_3_norm1_weight,
    p_transformer_layers_3_norm1_bias,
    p_transformer_layers_3_linear1_weight,
    p_transformer_layers_3_linear1_bias,
    p_transformer_layers_3_linear2_weight,
    p_transformer_layers_3_linear2_bias,