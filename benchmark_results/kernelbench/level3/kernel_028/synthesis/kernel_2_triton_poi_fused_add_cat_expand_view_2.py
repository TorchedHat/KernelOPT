import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ─── Fused: cat(cls_token, patch_emb) + pos_embedding ───────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_cat_add_kernel(
    patch_ptr,     # [B, N, D]  (N=196)
    B, N, D,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # offsets into [B, N+1, D]
    d = offsets % D
    seq = (offsets // D) % (N + 1)
    b = offsets // (D * (N + 1))

    # cls token region: seq == 0
    is_cls = seq == 0
    # patch region: seq >= 1
    patch_idx = b * N * D + (seq - 1) * D + d

    cls_val = tl.load(cls_ptr + d, mask & is_cls, other=0.0)
    patch_val = tl.load(patch_ptr + patch_idx, mask & ~is_cls, other=0.0)
    src_val = tl.where(is_cls, cls_val, patch_val)

    pos_val = tl.load(pos_ptr + seq * D + d, mask, other=0.0)
    out_val = src_val + pos_val
    tl.store(out_ptr + offsets, out_val, mask)


# ─── Fused LayerNorm ─────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    out_row = out_ptr + row * N

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    # Normalize and scale
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_row + cols, y.to(tl.float32), mask=mask)


# ─── Fused: add residual + LayerNorm ─────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, residual_ptr, w_ptr, b_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    res_row = residual_ptr + row * N
    out_row = out_ptr + row * N

    # Compute mean of (x + residual)
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x + r, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        diff = (x + r) - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    # Normalize
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = ((x + r) - mean) * rstd * w + b
        tl.store(out_row + cols, y.to(tl.float32), mask=mask)


# ─── Fused: bias_add + ReLU ──────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    total_elements, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    d = offsets % D
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + d, mask=mask, other=0.0)
    y = x + b
    y = tl.where(y > 0, y, tl.zeros_like(y))
    tl.store(out_ptr + offsets, y, mask=mask)


# ─── Fused: bias_add (for linear output) ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_bias_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    total_elements, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    d = offsets % D
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + d, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + b, mask=mask)


# ─── Fused: bias_add + GELU ──────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_bias_gelu_kernel(
    x_ptr, bias_ptr, out_ptr,
    total_elements, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    d = offsets % D
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + d, mask=mask, other=0.0).to(tl.float32)
    y = x + b
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    y_gelu = 0.5 * y * (1.0 + libdevice.tanh(c * (y + 0.044715 * y * y * y)))
    tl.store(out_ptr + offsets, y_gelu, mask=mask)


# ─── Patch unfold kernel ──────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def patch_unfold_kernel(
    img_ptr, out_ptr,
    B, C, H, W,
    patch_size,
    num_patches_h, num_patches_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    patch_dim = C * patch_size * patch_size
    num_patches = num_patches_h * num_patches_w

    # offsets into [B, num_patches, patch_dim]
    d = offsets % patch_dim
    p = (offsets // patch_dim) % num_patches
    b = offsets // (patch_dim * num_patches)

    # decode d into (c, ph, pw)
    pw = d % patch_size
    ph = (d // patch_size) % patch_size
    c = d // (patch_size * patch_size)

    # decode p into (py, px)
    px = p % num_patches_w
    py = p // num_patches_w

    # image coordinates
    ih = py * patch_size + ph
    iw = px * patch_size + pw

    img_idx = b * (C * H * W) + c * (H * W) + ih * W + iw
    val = tl.load(img_ptr + img_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    """Apply layer norm using Triton kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        M, N, eps,
    )
    return out.reshape(orig_shape)


def triton_add_layernorm(x, residual, weight, bias, eps=1e-5):
    """Fused add + layer norm."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    res_2d = residual.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    fused_add_layernorm_kernel[grid](
        x_2d, res_2d, weight, bias, out,
        M, N, eps,
    )
    return out.reshape(orig_shape)


def triton_bias_relu(x, bias):
    """Fused bias add + ReLU."""
    out = torch.empty_like(x)
    total = x.numel()
    D = bias.shape[0]
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fused_bias_relu_kernel[grid](x, bias, out, total, D)
    return out


def triton_bias_add(x, bias):
    """Fused bias add."""
    out = torch.empty_like(x)
    total = x.numel()
    D = bias.shape[0]
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fused_bias_add_kernel[grid](x, bias, out, total, D)
    return out


def triton_bias_gelu(x, bias):
    """Fused bias add + GELU."""
    out = torch.empty_like(x)
    total = x.numel()
    D = bias.shape[0]
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fused_bias_gelu_kernel[grid](x, bias, out, total, D)
    return out


def transformer_layer(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                       norm1_weight, norm1_bias, linear1_weight, linear1_bias,
                       linear2_weight, linear2_bias, norm2_weight, norm2_bias,
                       num_heads=8, dropout=0.0):
    """Single transformer encoder layer."""
    B, S, D = x.shape
    head_dim = D // num_heads

    # Self-attention
    # QKV projection: [B*S, D] @ [D, 3D]^T + bias
    x_2d = x.reshape(B * S, D)
    qkv = torch.mm(x_2d, in_proj_weight.t())
    qkv = triton_bias_add(qkv, in_proj_bias)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)

    q = qkv[:, :, 0].transpose(1, 2)  # [B, heads, S, head_dim]
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)

    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    attn_out = torch.matmul(attn, v)  # [B, heads, S, head_dim]

    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B * S, D)
    attn_out = torch.mm(attn_out, out_proj_weight.t())
    attn_out = triton_bias_add(attn_out, out_proj_bias)
    attn_out = attn_out.reshape(B, S, D)

    # Residual + LayerNorm 1
    x = triton_add_layernorm(attn_out, x, norm1_weight, norm1_bias)

    # FFN
    x_2d = x.reshape(B * S, D)
    ff = torch.mm(x_2d, linear1_weight.t())
    ff = triton_bias_relu(ff, linear1_bias)
    ff = torch.mm(ff, linear2_weight.t())
    ff = triton_bias_add(ff, linear2_bias)
    ff = ff.reshape(B, S, D)

    # Residual + LayerNorm 2
    x = triton_add_layernorm(ff, x, norm2_weight, norm2_bias)

    return x


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
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = C * patch_size * patch_size
    D = 512

    # Step 1: Extract patches using Triton kernel
    patches = torch.empty(B, num_patches, patch_dim, device=img.device, dtype=img.dtype)
    total = B * num_patches * patch_dim
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    patch_unfold_kernel[grid](
        img, patches,
        B, C, H, W,
        patch_size, num_patches_h, num_patches_w,
        total,
    )

    # Step 2: Patch embedding: [B, num_patches, patch_dim] @ [patch_dim, D]^T + bias
    patches_2d = patches.reshape(B * num_patches, patch_dim)
    emb = torch.mm(patches_2d, p_patch_to_embedding_weight.t())
    emb = triton_bias_add(emb, p_patch_to_embedding_bias)
    emb = emb.reshape(B, num_patches, D)

    # Step 3: Cat cls_token + patch embeddings + add positional embedding
    x = torch.empty(B, num_patches + 1, D, device=img.device, dtype=img.dtype)
    total_x = B * (num_patches + 1) * D
    grid2 = lambda meta: (triton.cdiv(total_x, meta['BLOCK_SIZE']),)
    fused_cat_add_kernel[grid2](
        p_cls_token,
        emb,
        p_pos_embedding,
        x,
        B, num_patches, D,
        total_x,
    )

    # Step 4: Transformer layers
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
        (p_transformer_layers_2_self_attn_in_proj_weight, p_transformer_layers_2_self_attn_in_proj_bias,
         p_transformer_layers_2_self_attn_out_proj_weight, p_transformer_layers_2_self_attn_out_proj_bias,
         p_transformer_layers_2_norm1_weight, p_transformer_layers_2_norm1_bias,
         p_transformer_layers_2_linear1_weight, p_transformer_layers_2_linear1_bias,
         p_transformer_layers_2_linear2_weight, p_transformer_layers_2_linear2_bias,
         p_transformer_layers_2_norm2_weight, p_transformer_layers_2_norm2_bias),
        (p_transformer_layers_3_self_attn_in_proj_weight, p_transformer_layers_3_self_attn_in_proj_bias,
         p_transformer_layers_3_self_attn_out_proj_weight, p_transformer_layers_3_self_attn_out_proj_bias,
         p_transformer_layers_3_norm1_weight, p_transformer_layers_3_norm1_bias,
         p_transformer_layers_3_linear1_weight, p_transformer_layers_3_linear1_bias,
         p_transformer_layers_3_linear2_weight, p_transformer_layers_3_linear2_bias,
         p_transformer_layers_3_norm2_weight, p_transformer_layers_3_norm2_bias),
        (p_transformer_layers_4_self_attn_in_proj_weight, p_transformer_layers_4_self_attn_in_proj_bias,
         p_transformer_layers_4_self_attn_out_proj_weight, p_transformer_layers_4_self_attn_out_proj_bias,
         p_transformer_layers_4_norm1_weight, p_transformer_layers_4_norm1_bias,
         p_transformer_layers_4_linear1_weight, p_transformer_layers_4_linear1_bias,
         p_transformer_layers_4_linear2_weight, p_transformer_layers_4_linear2_bias,