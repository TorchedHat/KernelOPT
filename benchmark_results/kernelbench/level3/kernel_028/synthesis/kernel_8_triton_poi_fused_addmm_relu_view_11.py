import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
    rstd = 1.0 / libdevice.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](x_2d, weight, bias, out, M, N, eps)
    return out.reshape(orig_shape)


# ============================================================
# Fused ReLU kernel
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
def relu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, N)
    return out


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
# Fused patch embedding + cat + add kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['total'],
)
@triton.jit
def cat_add_kernel(
    cls_ptr, patch_ptr, pos_ptr, out_ptr,
    B, seq_len, dim,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < B * seq_len * dim

    # Decode indices
    d = offs % dim
    s = (offs // dim) % seq_len
    b = offs // (seq_len * dim)

    # Load pos embedding
    pos = tl.load(pos_ptr + s * dim + d, mask=mask, other=0.0)

    # Load cls or patch
    is_cls = s == 0
    cls_val = tl.load(cls_ptr + d, mask=mask & is_cls, other=0.0)
    patch_val = tl.load(patch_ptr + b * (seq_len - 1) * dim + (s - 1) * dim + d,
                        mask=mask & ~is_cls, other=0.0)
    val = tl.where(is_cls, cls_val, patch_val)
    out = val + pos
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Linear (matmul) using tl.dot
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
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
        acc = acc + bias[None, :]

    tl.store(out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :], acc, mask=out_mask)


def triton_linear(x, weight, bias=None):
    """x: (M, K), weight: (N, K) -> out: (M, N)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x_2d, weight, bias if bias is not None else x_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return out.reshape(*orig_shape[:-1], N)


# ============================================================
# Attention helper using scaled_dot_product_attention
# ============================================================
def multi_head_attention(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
    """
    """
    B, S, D = x.shape
    head_dim = embed_dim // num_heads

    # Project to Q, K, V
    qkv = triton_linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3*D)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    q = qkv[:, :, 0].transpose(1, 2)  # (B, H, S, head_dim)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)

    # Scaled dot product attention
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)  # (B, H, S, head_dim)

    out = out.transpose(1, 2).contiguous().reshape(B * S, D)
    out = triton_linear(out, out_proj_weight, out_proj_bias)
    return out.reshape(B, S, D)


# ============================================================
# Transformer encoder layer
# ============================================================
def transformer_encoder_layer(x, layer_params):
    """
    """
    in_proj_weight = layer_params['in_proj_weight']
    in_proj_bias = layer_params['in_proj_bias']
    out_proj_weight = layer_params['out_proj_weight']
    out_proj_bias = layer_params['out_proj_bias']
    norm1_weight = layer_params['norm1_weight']
    norm1_bias = layer_params['norm1_bias']
    linear1_weight = layer_params['linear1_weight']
    linear1_bias = layer_params['linear1_bias']
    linear2_weight = layer_params['linear2_weight']
    linear2_bias = layer_params['linear2_bias']
    norm2_weight = layer_params['norm2_weight']
    norm2_bias = layer_params['norm2_bias']
    num_heads = layer_params['num_heads']
    embed_dim = layer_params['embed_dim']

    # Self-attention with residual
    attn_out = multi_head_attention(x, in_proj_weight, in_proj_bias,
                                     out_proj_weight, out_proj_bias,
    x = x + attn_out
    x = triton_layer_norm(x, norm1_weight, norm1_bias)

    # FFN with residual
    ffn = triton_linear(x, linear1_weight, linear1_bias)
    ffn = triton_relu(ffn)
    ffn = triton_linear(ffn, linear2_weight, linear2_bias)
    x = x + ffn
    x = triton_layer_norm(x, norm2_weight, norm2_bias)

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
    embed_dim = 512

    # Patch embedding
    p = patch_size
    # Unfold to get patches: (B, C, H//p, W//p, p, p)
    x = img.unfold(2, p, p).unfold(3, p, p)
    # x shape: (B, C, H//p, W//p, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    # x shape: (B, H//p, W//p, C, p, p)
    x = x.reshape(B, -1, C * p * p)
    # x shape: (B, num_patches, patch_dim)

    # Linear projection
    x = triton_linear(x, p_patch_to_embedding_weight, p_patch_to_embedding_bias)
    # x shape: (B, num_patches, embed_dim)

    # Prepend cls token and add positional embedding
    cls_tokens = p_cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
    x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
    x = x + p_pos_embedding  # (B, num_patches+1, embed_dim)

    # Transformer layers
    layer_params_list = [
        {
            'in_proj_weight': p_transformer_layers_0_self_attn_in_proj_weight,
            'in_proj_bias': p_transformer_layers_0_self_attn_in_proj_bias,
            'out_proj_weight': p_transformer_layers_0_self_attn_out_proj_weight,
            'out_proj_bias': p_transformer_layers_0_self_attn_out_proj_bias,
            'norm1_weight': p_transformer_layers_0_norm1_weight,
            'norm1_bias': p_transformer_layers_0_norm1_bias,
            'linear1_weight': p_transformer_layers_0_linear1_weight,
            'linear1_bias': p_transformer_layers_0_linear1_bias,
            'linear2_weight': p_transformer_layers_0_linear2_weight,
            'linear2_bias': p_transformer_layers_0_linear2_bias,
            'norm2_weight': p_transformer_layers_0_norm2_weight,
            'norm2_bias': p_transformer_layers_0_norm2_bias,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
        },
        {
            'in_proj_weight': p_transformer_layers_1_self_attn_in_proj_weight,
            'in_proj_bias': p_transformer_layers_1_self_attn_in_proj_bias,
            'out_proj_weight': p_transformer_layers_1_self_attn_out_proj_weight,
            'out_proj_bias': p_transformer_layers_1_self_attn_out_proj_bias,
            'norm1_weight': p_transformer_layers_1_norm1_weight,
            'norm1_bias': p_transformer_layers_1_norm1_bias,
            'linear1_weight': p_transformer_layers_1_linear1_weight,
            'linear1_bias': p_transformer_layers_1_linear1_bias,
            'linear2_weight': p_transformer_layers_1_linear2_weight,
            'linear2_bias': p_transformer_layers_1_linear2_bias,
            'norm2_weight': p_transformer_layers_1_norm2_weight,
            'norm2_bias': p_transformer_layers_1_norm2_bias,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
        },
        {
            'in_proj_weight': p_transformer_layers_2_self_attn_in_proj_weight,
            'in_proj_bias': p_transformer_layers_2_self_attn_in_proj_bias,
            'out_proj_weight': p_transformer_layers_2_self_attn_out_proj_weight,
            'out_proj_bias': p_transformer_layers_2_self_attn_out_proj_bias,
            'norm1_weight': p_transformer_layers_2_norm1_weight,
            'norm1_bias': p_transformer_layers_2_norm1_bias,
            'linear1_weight': p_transformer_layers_2_linear1_weight,
            'linear1_bias': p_transformer_layers_2_linear1_bias,
            'linear2_weight': p_transformer_layers_2_linear2_weight,
            'linear2_bias': p_transformer_layers_2_linear2_bias,
            'norm2_weight': p_transformer_layers_2_norm2_weight,
            'norm2_bias': p_transformer_layers_2_norm2_bias,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
        },
        {
            'in_proj_weight': p_transformer_layers_3_self_attn_in_proj_weight,
            'in_proj_bias': p_transformer_layers_3_self_attn_in_proj_bias,
            'out_proj_weight': p_transformer_layers_3_self_attn_out_proj_weight,
            'out_proj_bias': p_transformer_layers_3_self_attn_out_proj_bias,
            'norm1_weight': p_transformer_layers_3_norm1_weight,
            'norm1_bias': p_transformer_layers_3_norm1_bias,
            'linear1_weight': p_transformer_layers_3_linear1_weight,
            'linear1_bias': p_transformer_layers_3_linear1_bias,
            'linear2_weight': p_transformer_layers_3_linear2_weight,
            'linear2_bias': p_transformer_layers_3_linear2_bias,
            'norm2_weight': p_transformer_layers_3_norm2_weight,
            'norm2_bias': p_transformer_layers_3_norm2_bias,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
        },
        {
            'in_proj_weight': p_transformer_layers_4_self_attn_in_proj_weight,
            'in_proj_bias': p_transformer_layers_4_self_attn_in_proj_bias,
            'out_proj_weight': p_transformer_layers_4_self_attn_out_proj_weight,
            'out_proj_bias': p_transformer_layers_4_self_attn_out_proj_bias,
            'norm1_weight': p_transformer_layers_4_norm1_weight,
            'norm1_bias': p_transformer_layers_4_norm1_bias,
            'linear1_weight': p_transformer_layers_4_linear1_weight,
            'linear1_bias': p_transformer_layers_4_linear1_bias,
            'linear2_weight': p_transformer_layers_4_linear2_weight,
            'linear2_bias': p_transformer_layers_4_linear2_bias,
            'norm2_weight': p_transformer_layers_4_norm2_weight,
            'norm2_bias': p_transformer_layers_4_norm2_bias,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
        },
        {
            'in_proj_weight': p_transformer_layers_5_self_attn_in_proj_weight,
            'in_proj_bias': p_transformer_layers_5_self_attn_in_proj_bias,
            'out_proj_weight': p_transformer_layers_5_self_attn_out_proj_weight,
            'out_proj_bias': p_transformer_layers_5_self_attn_out_proj_bias,
            'norm1_weight': p_transformer_layers_5_norm1_weight,
            'norm1_bias': p_transformer_layers_5_norm1_bias,
            'linear1_weight': p_transformer_layers_5_linear1_weight,
            'linear1_bias': p_transformer_layers_5_linear1_bias,
            'linear2_weight': p_transformer_layers_5_linear2_weight,
            'linear2_bias': p_transformer_layers_5_linear2_bias,
            'norm2_weight': p_transformer_layers_5_norm2_weight,
            'norm2_bias': p_transformer_layers_5_norm2_bias,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
        },
    ]

    for layer_params in layer_params_list:
        x = transformer_encoder_layer(x, layer_params)

    # Extract CLS token
    cls_out = x[:, 0]  # (B, embed_dim)

    # MLP head
    out = triton_linear(cls_out, p_mlp_head_0_weight, p_mlp_head_0_bias)
    out = triton_gelu(out)
    out = triton_linear(out, p_mlp_head_3_weight, p_mlp_head_3_bias)

    return out


def get_inputs():
    image_size = 224
    patch_size = 16
    num_classes = 10
    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 2048
    channels = 3

    p_patch_to_embedding_weight = torch.randn(512, 768, device='cuda')
    p_patch_to_embedding_bias = torch.randn(512, device='cuda')
    p_cls_token = torch.randn(1, 1, 512, device='cuda')
    p_pos_embedding = torch.randn(1, 197, 512,