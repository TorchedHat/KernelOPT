import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N

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


def layer_norm_triton(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](x_2d, weight, bias, out, M, N, eps)
    return out.reshape(orig_shape)


# ============================================================
# Fused GEMM kernel (M x K) @ (K x N) + bias -> ReLU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    DO_RELU: tl.constexpr,
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
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def linear_triton(x, weight, bias=None, do_relu=False):
    """Compute x @ weight.T + bias"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    gemm_relu_kernel[grid](
        x_2d, weight, bias if bias is not None else x_2d,
        out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N, K), so stride_bk=weight.stride(1), stride_bn=weight.stride(0)
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
        DO_RELU=do_relu,
    )
    
    new_shape = orig_shape[:-1] + (N,)
    return out.reshape(new_shape)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
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
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def gelu_triton(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# ============================================================
# Patch embedding kernel: unfold + reshape + linear
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def patch_embed_kernel(
    img_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, H, W,
    patch_size,
    num_patches_h, num_patches_w,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    weight: (N, K) where K = C * patch_size * patch_size, N = dim
    out: (B, num_patches, N) reshaped to (M, N) where M = B * num_patches
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Each row in M corresponds to (batch, patch_h, patch_w)
    # offs_m: batch * num_patches_h * num_patches_w + ph * num_patches_w + pw
    num_patches = num_patches_h * num_patches_w

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        # K = C * patch_size * patch_size
        # offs_k -> (c, kh, kw)
        kw = offs_k % patch_size
        kh = (offs_k // patch_size) % patch_size
        c = offs_k // (patch_size * patch_size)

        # offs_m -> (batch, ph, pw)
        pw = offs_m % num_patches_w
        ph = (offs_m // num_patches_w) % num_patches_h
        batch = offs_m // num_patches

        # Input pixel coordinates
        ih = ph[:, None] * patch_size + kh[None, :]
        iw = pw[:, None] * patch_size + kw[None, :]
        b = batch[:, None]
        ch = c[None, :]

        valid_m = offs_m < M
        valid_k = offs_k < K

        img_idx = b * (C * H * W) + ch * (H * W) + ih * W + iw
        a = tl.load(img_ptr + img_idx, mask=valid_m[:, None] & valid_k[None, :], other=0.0)

        # weight is (N, K), we need (K, N) for the dot
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w = tl.load(weight_ptr + w_idx, mask=(offs_n[None, :] < N) & valid_k[:, None], other=0.0)

        acc = tl.dot(a, w, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


def patch_embed_triton(img, weight, bias, patch_size):
    B, C, H, W = img.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    K = C * patch_size * patch_size
    N = weight.shape[0]
    M = B * num_patches

    out = torch.empty((M, N), device=img.device, dtype=img.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    patch_embed_kernel[grid](
        img, weight, bias, out,
        B, C, H, W,
        patch_size,
        num_patches_h, num_patches_w,
        M, N, K,
    )

    return out.reshape(B, num_patches, N)


# ============================================================
# Fused add + LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def add_layer_norm_kernel(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    residual_ptr = residual_ptr + row * N
    out_ptr = out_ptr + row * N

    # Compute mean of (x + residual)
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(tl.where(mask, x + r, 0.0), axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + r - mean, 0.0)
        var += tl.sum(val * val, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x + r - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


def add_layer_norm_triton(x, residual, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    r_2d = residual.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    add_layer_norm_kernel[grid](x_2d, r_2d, weight, bias, out, M, N, eps)
    return out.reshape(orig_shape)


# ============================================================
# Multi-head self-attention using scaled_dot_product_attention
# ============================================================
def mha_triton(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
    """
    """
    B, S, D = x.shape
    
    # QKV projection
    qkv = linear_triton(x, in_proj_weight, in_proj_bias)  # (B, S, 3*D)
    
    # Split into Q, K, V
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    q = qkv[:, :, 0].transpose(1, 2)  # (B, num_heads, S, head_dim)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    
    # Scaled dot product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)  # (B, num_heads, S, head_dim)
    
    # Reshape
    out = out.transpose(1, 2).contiguous().reshape(B * S, D)
    
    # Output projection
    out = linear_triton(out, out_proj_weight, out_proj_bias)
    return out.reshape(B, S, D)


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
    # Self-attention
    attn_out = mha_triton(x, in_proj_weight, in_proj_bias,
                          out_proj_weight, out_proj_bias,
    
    # Add & Norm 1
    x = add_layer_norm_triton(attn_out, x, norm1_weight, norm1_bias)
    
    # FFN
    ffn_out = linear_triton(x, linear1_weight, linear1_bias, do_relu=True)
    ffn_out = linear_triton(ffn_out, linear2_weight, linear2_bias)
    
    # Add & Norm 2
    x = add_layer_norm_triton(ffn_out, x, norm2_weight, norm2_bias)
    
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
    p_transformer_layers_3_norm2_weight,
    p_transformer_layers_3_norm2_bias,
    p_transformer_layers_4_self_attn_in_proj_bias,
    p_transformer_layers_4_self_attn_in_proj_weight,
    p_transformer_layers_4_self_attn_out_proj_weight,
    p_transformer_layers_4_self_attn_out_proj_bias,
    p_transformer_layers_4_norm1_weight,
    p_transformer_layers_4_norm1_bias,
    p_transformer_layers_4_linear1_weight,
    p_transformer_layers_4_linear1_bias,
    p_transformer_layers_4_linear2_weight,
    p_transformer_layers_4_linear2_bias,
    p_transformer_layers_4_norm2_weight,
    p_transformer_layers_4_norm2_bias,
    p_transformer_layers_5_self_attn_in_proj_bias,
    p_transformer_layers_5_self_attn_in_proj_weight,
    p_transformer_layers_5_self_attn_out_proj_weight,
    p_transformer_layers_5_self_attn_out_proj_bias,
    p_transformer_layers_5_norm1_weight,
    p_transformer_layers_5_norm1_bias,
    p_transformer_layers_5_linear1_weight,
    p_transformer_layers_5_linear1_bias,
    p_transformer_layers_5_linear2_weight,
    p_transformer_layers_5_linear2_bias,
    p_transformer_layers_5_norm2_weight,
    p_transformer_layers_5_norm2_bias,
    p_mlp_head_0_weight,
    p_mlp_head_0_bias,
    p_mlp_head_3_weight,
    p_mlp_head_3_bias,
    img,
):
    B, C, H, W = img.shape
    patch_size = 16
    num_heads = 8
    head_dim = 64
    
    # Patch embedding
    x = patch_embed_triton(img, p_patch_to_embedding_weight, p_patch_to_embedding_bias, patch_size)
    # x: (B, 196, 512)
    
    # Prepend CLS token
    cls_tokens = p_cls_token.expand(B, -1, -1)  # (B, 1, 512)
    x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 512)
    
    # Add positional embedding
    x = x + p_pos_embedding  # (B, 197, 512)
    
    # Transformer layers
    layer_params = [
        (p_transformer_layers_0_self_attn_in_proj_weight, p_transformer_layers_0_self_attn_in_proj_bias,
         p_transformer_layers_0_self_attn_out_proj_weight, p_transformer_layers_0_self_attn_out_proj_bias,
         p_transformer_layers_0_norm1_weight, p_transformer_layers_0_norm1_bias,
         p_transformer_layers_0_linear1_weight, p_transformer_layers_0_linear1_bias,
         p_transformer_layers_0_linear2_weight, p_transformer_layers_0_linear2_bias,
         p_transformer_layers_0_norm2_weight, p_transformer_layers_0_norm2_bias),
        (p_transformer_layers_1_self_attn_in_proj_weight, p_transformer_layers_1_self_attn_in_proj_bias,
         p_transformer_layers_1_self_attn_out_proj_weight, p_transformer_layers_1_self_attn_out_proj_bias,
         p_transformer_layers_1_norm1_weight, p_transformer_layers_1_norm1_bias,
         p_transformer_layers_1_linear1_weight, p_transformer_layers_1_linear1_bias,
         p_transformer_layers_1_linear2_weight, p_transformer_layers_1_linear2_bias,
         p_transformer_layers_1_norm2_weight, p_transformer_layers_1_norm2_bias),
        (p_transformer_layers_2_self_attn_in_proj_weight,