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
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


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


# ============================================================
# Fused patch embedding + cat + add pos embedding
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def cat_add_kernel(
    cls_ptr, patch_ptr, pos_ptr, out_ptr,
    B, N_patches, D,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one token in one batch)
    pid = tl.program_id(0)
    batch = pid // (N_patches + 1)
    token = pid % (N_patches + 1)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    if token == 0:
        # cls token
        x = tl.load(cls_ptr + offs, mask=mask, other=0.0)
    else:
        # patch embedding
        x = tl.load(patch_ptr + batch * N_patches * D + (token - 1) * D + offs, mask=mask, other=0.0)

    pos = tl.load(pos_ptr + token * D + offs, mask=mask, other=0.0)
    out = x + pos
    tl.store(out_ptr + pid * D + offs, out, mask=mask)


# ============================================================
# GEMM kernel (tiled matmul with tl.dot)
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
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def triton_matmul(a, b, bias=None):
    """Perform matrix multiplication a @ b.T + bias using Triton."""
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    HAS_BIAS = bias is not None
    if not HAS_BIAS:
        bias_ptr = a  # dummy
    else:
        bias_ptr = bias
    matmul_bias_kernel[grid](
        a, b, bias_ptr, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # b is transposed: stride_bk=b.stride(1), stride_bn=b.stride(0)
        c.stride(0), c.stride(1),
        HAS_BIAS=HAS_BIAS,
    )
    return c


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    layer_norm_kernel[(M,)](
        x_2d, weight, bias, out,
        M, N, eps,
        N=N,
    )
    return out.reshape(orig_shape)


def triton_relu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, N, N=N)
    return out


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N, N=N)
    return out


def transformer_layer(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                      norm1_weight, norm1_bias, linear1_weight, linear1_bias,
                      linear2_weight, linear2_bias, norm2_weight, norm2_bias,
                      num_heads=8):
    """Single transformer encoder layer."""
    B, S, D = x.shape
    head_dim = D // num_heads

    # Self-attention
    # in_proj: (B*S, D) @ (D, 3D) -> (B*S, 3D)
    x_2d = x.reshape(B * S, D)
    qkv = triton_matmul(x_2d, in_proj_weight, in_proj_bias)  # (B*S, 3D)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    q = qkv[:, :, 0].transpose(1, 2)  # (B, heads, S, head_dim)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)

    # Scaled dot product attention
    scale = math.sqrt(head_dim)
    attn = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn = torch.softmax(attn, dim=-1)
    attn_out = torch.matmul(attn, v)  # (B, heads, S, head_dim)
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B * S, D)

    # out_proj
    attn_out = triton_matmul(attn_out, out_proj_weight, out_proj_bias)
    attn_out = attn_out.reshape(B, S, D)

    # Residual + norm1
    x = x + attn_out
    x = triton_layer_norm(x, norm1_weight, norm1_bias)

    # FFN
    x_2d = x.reshape(B * S, D)
    ffn = triton_matmul(x_2d, linear1_weight, linear1_bias)  # (B*S, mlp_dim)
    ffn = triton_relu(ffn)
    ffn = triton_matmul(ffn, linear2_weight, linear2_bias)  # (B*S, D)
    ffn = ffn.reshape(B, S, D)

    # Residual + norm2
    x = x + ffn
    x = triton_layer_norm(x, norm2_weight, norm2_bias)

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

    # Patch extraction and embedding
    # img: (B, C, H, W) -> patches: (B, num_patches, patch_dim)
    x = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # x: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    # x: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    x = x.reshape(B, num_patches, patch_dim)
    # x: (B, num_patches, patch_dim)

    # Patch embedding: (B*num_patches, patch_dim) @ (patch_dim, D) + bias
    x_2d = x.reshape(B * num_patches, patch_dim)
    x_emb = triton_matmul(x_2d, p_patch_to_embedding_weight, p_patch_to_embedding_bias)
    # x_emb: (B*num_patches, D)
    D = p_patch_to_embedding_bias.shape[0]
    x_emb = x_emb.reshape(B, num_patches, D)

    # Prepend cls token and add positional embedding
    cls_tokens = p_cls_token.expand(B, -1, -1)  # (B, 1, D)
    x = torch.cat([cls_tokens, x_emb], dim=1)  # (B, num_patches+1, D)
    x = x + p_pos_embedding  # (B, num_patches+1, D)

    # Transformer layers
    layer_params = [
        (p_transformer_layers_0_self_attn_in_proj_weight,
         p_transformer_layers_0_self_attn_in_proj_bias,
         p_transformer_layers_0_self_attn_out_proj_weight,
         p_transformer_layers_0_self_attn_out_proj_bias,
         p_transformer_layers_0_norm1_weight,
         p_transformer_layers_0_norm1_bias,
         p_transformer_layers_0_linear1_weight,
         p_transformer_layers_0_linear1_bias,
         p_transformer_layers_0_linear2_weight,
         p_transformer_layers_0_linear2_bias,
         p_transformer_layers_0_norm2_weight,
         p_transformer_layers_0_norm2_bias),
        (p_transformer_layers_1_self_attn_in_proj_weight,
         p_transformer_layers_1_self_attn_in_proj_bias,
         p_transformer_layers_1_self_attn_out_proj_weight,
         p_transformer_layers_1_self_attn_out_proj_bias,
         p_transformer_layers_1_norm1_weight,
         p_transformer_layers_1_norm1_bias,
         p_transformer_layers_1_linear1_weight,
         p_transformer_layers_1_linear1_bias,
         p_transformer_layers_1_linear2_weight,
         p_transformer_layers_1_linear2_bias,
         p_transformer_layers_1_norm2_weight,
         p_transformer_layers_1_norm2_bias),
        (p_transformer_layers_2_self_attn_in_proj_weight,
         p_transformer_layers_2_self_attn_in_proj_bias,
         p_transformer_layers_2_self_attn_out_proj_weight,
         p_transformer_layers_2_self_attn_out_proj_bias,
         p_transformer_layers_2_norm1_weight,
         p_transformer_layers_2_norm1_bias,
         p_transformer_layers_2_linear1_weight,
         p_transformer_layers_2_linear1_bias,
         p_transformer_layers_2_linear2_weight,
         p_transformer_layers_2_linear2_bias,
         p_transformer_layers_2_norm2_weight,
         p_transformer_layers_2_norm2_bias),
        (p_transformer_layers_3_self_attn_in_proj_weight,
         p_transformer_layers_3_self_attn_in_proj_bias,
         p_transformer_layers_3_self_attn_out_proj_weight,
         p_transformer_layers_3_self_attn_out_proj_bias,
         p_transformer_layers_3_norm1_weight,
         p_transformer_layers_3_norm1_bias,
         p_transformer_layers_3_linear1_weight,
         p_transformer_layers_3_linear1_bias,
         p_transformer_layers_3_linear2_weight,
         p_transformer_layers_3_linear2_bias,
         p_transformer_layers_3_norm2_weight,
         p_transformer_layers_3_norm2_bias),
        (p_transformer_layers_4_self_attn_in_proj_weight,
         p_transformer_layers_4_self_attn_in_proj_bias,
         p_transformer_layers_4_self_attn_out_proj_weight,
         p_transformer_layers_4_self_attn_out_proj_bias,
         p_transformer_layers_4_norm1_weight,
         p_transformer_layers_4_norm1_bias,
         p_transformer_layers_4_linear1_weight,
         p_transformer_layers_4_linear1_bias,
         p_transformer_layers_4_linear2_weight,
         p_transformer_layers_4_linear2_bias,
         p_transformer_layers_4_norm2_weight,
         p_transformer_layers_4_norm2_bias),
        (p_transformer_layers_5_self_attn_in_proj_weight,
         p_transformer_layers_5_self_attn_in_proj_bias,
         p_transformer_layers_5_self_attn_out_proj_weight,
         p_transformer_layers_5_self_attn_out_proj_bias,
         p_transformer_layers_5_norm1_weight,
         p_transformer_layers_5_norm1_bias,
         p_transformer_layers_5_linear1_weight,
         p_transformer_layers_5_linear1_bias,
         p_transformer_layers_5_linear2_weight,
         p_transformer_layers_5_linear2_bias,
         p_transformer_layers_5_norm2_weight,
         p_transformer_layers_5_norm2_bias),
    ]

    for params in layer_params:
        (in_proj_w, in_proj_b, out_proj_w, out_proj_b,
         norm1_w, norm1_b, lin1_w, lin1_b, lin2_w, lin2_b,
         norm2_w, norm2_b) = params
        x = transformer_layer(
            x, in_proj_w, in_proj_b, out_proj_w, out_proj_b,
            norm1_w, norm1_b, lin1_w, lin1_b, lin2_w, lin2_b,
            norm2_w, norm2_b,
            num_heads=8,
        )

    # Extract cls token
    cls_out = x[:, 0]  # (B, D)

    # MLP head: Linear -> GELU -> Linear
    mlp1 = triton_matmul(cls_out, p_mlp_head_0_weight, p_mlp_head_0_bias)  # (B, mlp_dim)
    mlp1 = triton_gelu(mlp1)
    out = triton_matmul(mlp1, p_mlp_head_3_weight, p_mlp_head_3_bias)  # (B, num_classes)

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

    p_patch_to_embedding_weight = torch.randn(dim, channels * patch_size * patch_size, device='cuda')
    p_patch_to_embedding_bias = torch.randn(dim, device='cuda')
    p_cls_token = torch.randn(1, 1, dim, device='cuda')
    p_pos_embedding = torch.randn(1, (image_size // patch_size) ** 2 + 1, dim, device='cuda')

    layer_params = []
    for _ in range(depth):
        layer_params.extend([
            torch.randn(3 * dim, device='cuda'),           # in_proj_bias
            torch.randn(3 * dim, dim, device='cuda'),      # in_proj_weight
            torch.randn(dim, dim, device='cuda'),          # out_proj_weight
            torch.randn(dim, device='cuda'),               # out_proj_bias
            torch.randn(dim, device='cuda'),               # norm1_weight
            torch.randn(dim, device='cuda'),               # norm1_bias
            torch.randn(mlp_dim, dim, device='cuda'),      # linear1_weight
            torch.randn(mlp_dim, device='cuda'),           # linear1_bias
            torch.randn(dim, mlp_dim, device='cuda'),      # linear2_weight
            torch.randn(dim, device='cuda'),               # linear2_bias
            torch.randn(dim, device='cuda'),               # norm2_weight
            torch.randn(dim, device='cuda'),               # norm2_bias
        ])

    p_mlp_head_0_weight = torch.randn(mlp_dim, dim, device='cuda')
    p_mlp_head_0_bias = torch.randn(mlp_dim, device='cuda')
    p_mlp_head_3_weight = torch.randn(num_classes