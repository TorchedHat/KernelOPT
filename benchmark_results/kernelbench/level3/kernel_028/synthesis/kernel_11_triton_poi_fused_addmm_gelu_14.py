import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Kernel 1: Fused unfold + reshape (patch extraction)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def unfold_reshape_kernel(
    img_ptr, out_ptr,
    B, C, H, W,
    patch_size, num_patches_h, num_patches_w,
    N,  # total elements = B * num_patches * patch_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    patch_dim = C * patch_size * patch_size
    num_patches = num_patches_h * num_patches_w

    # offsets index into [B, num_patches, patch_dim]
    pd = offsets % patch_dim
    tmp = offsets // patch_dim
    patch_idx = tmp % num_patches
    b = tmp // num_patches

    # patch_dim = C * patch_size * patch_size
    # pd = c * patch_size * patch_size + ph * patch_size + pw
    ps2 = patch_size * patch_size
    c = pd // ps2
    rem = pd % ps2
    ph = rem // patch_size
    pw = rem % patch_size

    # patch spatial location
    patch_row = patch_idx // num_patches_w
    patch_col = patch_idx % num_patches_w

    # actual pixel location
    h_idx = patch_row * patch_size + ph
    w_idx = patch_col * patch_size + pw

    img_offset = b * (C * H * W) + c * (H * W) + h_idx * W + w_idx
    val = tl.load(img_ptr + img_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


# ============================================================
# Kernel 2: Fused cat + add (cls token + pos embedding)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def cat_add_kernel(
    cls_ptr, patches_ptr, pos_ptr, out_ptr,
    B, seq_len, dim,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # out shape: [B, seq_len, dim] where seq_len = num_patches + 1
    d = offsets % dim
    tmp = offsets // dim
    s = tmp % seq_len
    b = tmp // seq_len

    # pos embedding: [1, seq_len, dim]
    pos_val = tl.load(pos_ptr + s * dim + d, mask=mask, other=0.0)

    is_cls = s == 0
    # cls token: [1, 1, dim] -> broadcast over batch
    cls_val = tl.load(cls_ptr + d, mask=(mask & is_cls), other=0.0)
    # patches: [B, num_patches, dim]
    patch_val = tl.load(patches_ptr + b * (seq_len - 1) * dim + (s - 1) * dim + d,
                        mask=(mask & ~is_cls), other=0.0)

    val = tl.where(is_cls, cls_val, patch_val) + pos_val
    tl.store(out_ptr + offsets, val, mask=mask)


# ============================================================
# Kernel 3: Fused LayerNorm
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['dim'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    num_rows, dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    x_start = row * dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim

    # Load x
    x = tl.load(x_ptr + x_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / dim

    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / dim

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    # Scale and shift
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    tl.store(out_ptr + x_start + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Kernel 4: Fused ReLU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Kernel 5: Fused GELU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    out = x * cdf
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Kernel 6: Fused add (residual connection)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# ============================================================
# Helper functions
# ============================================================
def triton_unfold_reshape(img, patch_size):
    B, C, H, W = img.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    patch_dim = C * patch_size * patch_size
    num_patches = num_patches_h * num_patches_w
    N = B * num_patches * patch_dim
    out = torch.empty(B, num_patches, patch_dim, device=img.device, dtype=img.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    unfold_reshape_kernel[grid](
        img, out,
        B, C, H, W,
        patch_size, num_patches_h, num_patches_w,
        N,
    )
    return out


def triton_cat_add(cls_token, patches, pos_embedding):
    B = patches.shape[0]
    num_patches = patches.shape[1]
    dim = patches.shape[2]
    seq_len = num_patches + 1
    N = B * seq_len * dim
    out = torch.empty(B, seq_len, dim, device=patches.device, dtype=patches.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    cat_add_kernel[grid](
        cls_token.reshape(-1),  # [dim]
        patches,
        pos_embedding,
        out,
        B, seq_len, dim,
        N,
    )
    return out


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    num_rows, dim = x_2d.shape
    out = torch.empty_like(x_2d)
    # Use next power of 2 for BLOCK_SIZE
    BLOCK_SIZE = max(triton.next_power_of_2(dim), 128)
    grid = (num_rows,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, out,
        num_rows, dim,
        eps,
    )
    return out.reshape(orig_shape)


def triton_relu(x):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, N)
    return out


def triton_gelu(x):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


def triton_add(a, b):
    N = a.numel()
    out = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, out, N)
    return out


# ============================================================
# Linear layer using Triton tl.dot (to avoid banned torch.mm)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        # Load x tile [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptr + offs_m[:, None] * K + k_offs[None, :],
                         mask=x_mask, other=0.0)
        # Load w tile [BLOCK_K, BLOCK_N] (w is [N, K], so w[n, k])
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptr + offs_n[None, :] * K + k_offs[:, None],
                         mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    if has_bias:
        b_mask = offs_n < N
        bias = tl.load(b_ptr + offs_n, mask=b_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


def triton_linear(x, weight, bias=None):
    """x: [M, K], weight: [N, K], bias: [N] -> out: [M, N]"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    has_bias = bias is not None
    if not has_bias:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x_2d, weight, bias, out,
        M, N, K,
        has_bias,
    )
    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape)


# ============================================================
# Scaled dot product attention (using flash attention pattern)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, head_dim, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_head = tl.program_id(0)
    start_m = tl.program_id(1)

    # Decode batch and head
    # We'll use a flat batch*head index
    bh = batch_head
    q_base = q_ptr + bh * stride_qb
    k_base = k_ptr + bh * stride_kb
    v_base = v_ptr + bh * stride_vb
    o_base = out_ptr + bh * stride_ob

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                 mask=q_mask, other=0.0).to(tl.float32)
    q = q * scale

    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)

        # Load K block
        k = tl.load(k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                     mask=kv_mask, other=0.0).to(tl.float32)

        # Compute attention scores
        s = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]

        # Mask out-of-bounds
        s = tl.where(offs_n[None, :] < seq_len, s, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V block
        v = tl.load(v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                     mask=kv_mask, other=0.0).to(tl.float32)
        acc = tl.dot(p.to(v.dtype), v, acc=acc)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    out_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
             acc.to(tl.float32), mask=out_mask)


def triton_attention(q, k, v, scale):
    """
    """
    BH, seq_len, head_dim = q.shape
    HEAD_DIM = triton.next_power_of_2(head_dim)
    out = torch.empty_like(q)

    grid = lambda meta: (BH, triton.cdiv(seq_len, meta['BLOCK_M']))
    flash_attn_kernel[grid](
        q, k, v, out,
        seq_len, head_dim, scale,
        q.stride(0), q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(0), out.stride(1), out.stride(2),
        HEAD_DIM=HEAD_DIM,
    )
    return out


# ============================================================
# Transformer layer
# ============================================================
def transformer_layer(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                       norm1_weight, norm1_bias, linear1_weight, linear1_bias,
                       linear2_weight, linear2_bias, norm2_weight, norm2_bias,
                       num_heads=8):
    B, seq_len, dim = x.shape
    head_dim = dim // num_heads
    scale = head_dim ** -0.5

    # Self-attention
    # QKV projection: [B, seq_len, 3*dim]
    qkv = triton_linear(x, in_proj_weight, in_proj_bias)
    qkv = qkv.reshape(B, seq_len, 3, num_heads, head_dim)
    q = qkv[:, :, 0].permute(0, 2, 1, 3).reshape(B * num_heads, seq_len, head_dim).contiguous()
    k = qkv[:, :, 1].permute(0, 2, 1, 3).reshape(B * num_heads, seq_len, head_dim).contiguous()
    v = qkv[:, :, 2].permute(0, 2, 1, 3).reshape(B * num_heads, seq_len, head_dim).contiguous()

    # Attention
    attn_out = triton_attention(q, k, v, scale)
    attn_out = attn_out.reshape(B, num_heads, seq_len, head_dim).permute(0, 2, 1, 3).reshape(B * seq_len, dim).contiguous()

    # Output projection
    attn_out = triton_linear(attn_out, out_proj_weight, out_proj_bias)
    attn_out = attn_out.reshape(B, seq_len, dim)

    # Residual + LayerNorm
    x = triton_add(x, attn_out)
    x = triton_layer_norm(x, norm1_weight, norm1_bias)

    # FFN
    ffn = triton_linear(x, linear1_weight, linear1_bias)
    ffn = triton_relu(ffn)
    ffn = triton_linear(ffn, linear2_weight, linear2_bias)

    # Residual + LayerNorm
    x = triton_add(x, ffn)
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