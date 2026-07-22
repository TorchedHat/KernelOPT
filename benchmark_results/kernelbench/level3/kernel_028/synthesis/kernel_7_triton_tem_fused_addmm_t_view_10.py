import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton Kernels
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = X_ptr + row * stride_xm + offs * stride_xn
    mask = offs < N
    
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    x_norm = x_centered * rstd
    
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    y = x_norm * w + b
    
    y_ptrs = Y_ptr + row * stride_ym + offs * stride_yn
    tl.store(y_ptrs, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def patch_embed_unfold_kernel(
    img_ptr, out_ptr,
    B, C, H, W,
    patch_size,
    num_patches_h, num_patches_w,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    """Unfold image into patches: output shape [B, num_patches, C*patch_size*patch_size]"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total
    
    # N_total = B * num_patches_h * num_patches_w * C * patch_size * patch_size
    patch_dim = C * patch_size * patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Decode index
    idx = offs
    ps2 = patch_size * patch_size
    
    # out[b, ph*num_patches_w + pw, c*ps2 + ph_inner*patch_size + pw_inner]
    # idx = b * num_patches * patch_dim + patch_idx * patch_dim + feat_idx
    feat_idx = idx % patch_dim
    patch_idx = (idx // patch_dim) % num_patches
    b = idx // (patch_dim * num_patches)
    
    pw = patch_idx % num_patches_w
    ph = patch_idx // num_patches_w
    
    # feat_idx = c * ps2 + inner_h * patch_size + inner_w
    inner_w = feat_idx % patch_size
    inner_h = (feat_idx // patch_size) % patch_size
    c = feat_idx // ps2
    
    # Source pixel
    src_h = ph * patch_size + inner_h
    src_w = pw * patch_size + inner_w
    
    src_idx = b * (C * H * W) + c * (H * W) + src_h * W + src_w
    
    val = tl.load(img_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def cat_cls_add_pos_kernel(
    cls_ptr, patches_ptr, pos_ptr, out_ptr,
    B, seq_len, dim,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Concatenate cls token with patches, then add positional embedding"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # N = B * (1 + seq_len) * dim
    total_seq = 1 + seq_len
    
    d = offs % dim
    pos = (offs // dim) % total_seq
    b = offs // (total_seq * dim)
    
    is_cls = pos == 0
    
    cls_val = tl.load(cls_ptr + d, mask=mask & is_cls, other=0.0)
    patch_val = tl.load(patches_ptr + b * seq_len * dim + (pos - 1) * dim + d, 
                        mask=mask & ~is_cls, other=0.0)
    
    x = tl.where(is_cls, cls_val, patch_val)
    
    pos_val = tl.load(pos_ptr + pos * dim + d, mask=mask, other=0.0)
    
    out = x + pos_val
    tl.store(out_ptr + offs, out, mask=mask)


def triton_matmul(A, B, bias=None):
    """A: [M, K], B: [N, K] (weight matrix, will be transposed), bias: [N]"""
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    matmul_bias_kernel[grid](
        C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),  # B is [N, K], we want K stride and N stride
        C.stride(0), C.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return C


def triton_layer_norm(X, weight, bias, eps=1e-5):
    M, N = X.shape
    Y = torch.empty_like(X)
    
    layer_norm_kernel[(M,)](
        X, weight, bias, Y,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        eps,
        N=N,
    )
    return Y


def triton_relu(X):
    N = X.numel()
    Y = torch.empty_like(X)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](X, Y, N, N=N)
    return Y


def triton_gelu(X):
    N = X.numel()
    Y = torch.empty_like(X)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](X, Y, N, N=N)
    return Y


def triton_add(A, B):
    N = A.numel()
    C = torch.empty_like(A)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](A.reshape(-1), B.reshape(-1), C.reshape(-1), N, N=N)
    return C


def scaled_dot_product_attention_triton(q, k, v, scale):
    """
    """
    # Use flash attention style - but implement in triton
    # For correctness, implement standard attention
    B, S, D = q.shape
    
    # scores = q @ k.T / scale: [B, S, S]
    scores = triton_matmul_batched(q, k.transpose(1, 2)) * scale
    
    # softmax
    scores = softmax_triton(scores)
    
    # out = scores @ v: [B, S, D]
    out = triton_matmul_batched(scores, v)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    A_batch = A_ptr + batch * stride_ab
    B_batch = B_ptr + batch * stride_bb
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = A_batch + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_batch + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    C_batch = C_ptr + batch * stride_cb
    c_ptrs = C_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


def triton_matmul_batched(A, B):
    """A: [Batch, M, K], B: [Batch, K, N]"""
    Batch, M, K = A.shape
    _, _, N = B.shape
    C = torch.empty(Batch, M, N, device=A.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), Batch)
    
    batched_matmul_kernel[grid](
        A, B, C,
        Batch, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    return C


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel_2d(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x_ptrs = X_ptr + row * stride_xm + offs * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=float('-inf')).to(tl.float32)
    
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(tl.where(mask, exp_x, 0.0), axis=0)
    y = exp_x / sum_exp
    
    y_ptrs = Y_ptr + row * stride_ym + offs * stride_yn
    tl.store(y_ptrs, tl.where(mask, y, 0.0).to(tl.float32), mask=mask)


def softmax_triton(X):
    """X: [B, M, N] - apply softmax over last dim"""
    orig_shape = X.shape
    X_2d = X.reshape(-1, X.shape[-1])
    M, N = X_2d.shape
    Y_2d = torch.empty_like(X_2d)
    
    softmax_kernel_2d[(M,)](
        X_2d, Y_2d,
        M, N,
        X_2d.stride(0), X_2d.stride(1),
        Y_2d.stride(0), Y_2d.stride(1),
        N=N,
    )
    return Y_2d.reshape(orig_shape)


def transformer_layer(x, 
                      in_proj_weight, in_proj_bias,
                      out_proj_weight, out_proj_bias,
                      norm1_weight, norm1_bias,
                      linear1_weight, linear1_bias,
                      linear2_weight, linear2_bias,
                      norm2_weight, norm2_bias,
                      num_heads=8):
    """
    """
    B, S, D = x.shape
    head_dim = D // num_heads
    scale = head_dim ** -0.5
    
    # Self-attention
    # in_proj: [B*S, 3*D]
    x_2d = x.reshape(B * S, D)
    qkv = triton_matmul(x_2d, in_proj_weight, in_proj_bias)  # [B*S, 3*D]
    qkv = qkv.reshape(B, S, 3, D)
    
    q = qkv[:, :, 0, :].contiguous()  # [B, S, D]
    k = qkv[:, :, 1, :].contiguous()  # [B, S, D]
    v = qkv[:, :, 2, :].contiguous()  # [B, S, D]
    
    # Reshape for multi-head attention: [B*num_heads, S, head_dim]
    q = q.reshape(B, S, num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * num_heads, S, head_dim)
    k = k.reshape(B, S, num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * num_heads, S, head_dim)
    v = v.reshape(B, S, num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * num_heads, S, head_dim)
    
    # Attention
    attn_out = scaled_dot_product_attention_triton(q, k, v, scale)  # [B*num_heads, S, head_dim]
    
    # Reshape back
    attn_out = attn_out.reshape(B, num_heads, S, head_dim).permute(0, 2, 1, 3).reshape(B * S, D)
    
    # Out projection
    attn_out = triton_matmul(attn_out, out_proj_weight, out_proj_bias)  # [B*S, D]
    attn_out = attn_out.reshape(B, S, D)
    
    # Residual + LayerNorm
    x = x + attn_out
    x_2d = x.reshape(B * S, D)
    x_2d = triton_layer_norm(x_2d, norm1_weight, norm1_bias)
    x = x_2d.reshape(B, S, D)
    
    # FFN
    x_2d = x.reshape(B * S, D)
    ffn_out = triton_matmul(x_2d, linear1_weight, linear1_bias)  # [B*S, mlp_dim]
    ffn_out = triton_relu(ffn_out)
    ffn_out = triton_matmul(ffn_out, linear2_weight, linear2_bias)  # [B*S, D]
    ffn_out = ffn_out.reshape(B, S, D)
    
    # Residual + LayerNorm
    x = x + ffn_out
    x_2d = x.reshape(B * S, D)
    x_2d = triton_layer_norm(x_2d, norm2_weight, norm2_bias)
    x = x_2d.reshape(B, S, D)
    
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