import torch
import triton
import triton.language as tl
import math

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
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N
    
    # Compute mean
    mean = tl.zeros([], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros([], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)



import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
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
    x_row = x_ptr + row * N
    out_row = out_ptr + row * N

    mean = tl.zeros([], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    var = tl.zeros([], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_row + cols, out.to(tl.float32), mask=mask)


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
def relu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(out_ptr + offs, out, mask=mask)


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
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


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
def patch_embed_cat_add_kernel(
    cls_ptr, patch_ptr, pos_ptr, out_ptr,
    B, seq_len, dim,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # offs indexes into [B, seq_len, dim] flattened
    # seq_len = num_patches + 1 = 197
    d = offs % dim
    s = (offs // dim) % seq_len
    b = offs // (seq_len * dim)

    # cls token: s == 0, from cls_ptr[0, 0, d]
    # patch: s >= 1, from patch_ptr[b, s-1, d]
    is_cls = s == 0

    cls_val = tl.load(cls_ptr + d, mask=mask & is_cls, other=0.0)
    patch_idx = b * (seq_len - 1) * dim + (s - 1) * dim + d
    patch_val = tl.load(patch_ptr + patch_idx, mask=mask & ~is_cls, other=0.0)

    token_val = tl.where(is_cls, cls_val, patch_val)

    pos_val = tl.load(pos_ptr + s * dim + d, mask=mask, other=0.0)

    out = token_val + pos_val
    tl.store(out_ptr + offs, out, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](x_2d, weight, bias, out, M, N, eps)
    return out.reshape(orig_shape)


def triton_relu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, N)
    return out


def triton_gelu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


def transformer_layer(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                      norm1_weight, norm1_bias, linear1_weight, linear1_bias,
                      linear2_weight, linear2_bias, norm2_weight, norm2_bias,
                      num_heads=8):
    B, S, D = x.shape
    head_dim = D // num_heads

    # Self-attention
    # QKV projection
    x_2d = x.reshape(B * S, D)
    qkv = x_2d @ in_proj_weight.t() + in_proj_bias  # [B*S, 3*D]
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, S, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, S, head_dim]

    # Scaled dot product attention
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    attn_out = attn @ v  # [B, heads, S, head_dim]
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B * S, D)

    # Output projection
    attn_out = attn_out @ out_proj_weight.t() + out_proj_bias
    attn_out = attn_out.reshape(B, S, D)

    # Residual + norm1
    x = x + attn_out
    x = triton_layer_norm(x, norm1_weight, norm1_bias)

    # FFN
    x_2d = x.reshape(B * S, D)
    ffn = x_2d @ linear1_weight.t() + linear1_bias  # [B*S, mlp_dim]
    ffn = triton_relu(ffn)
    ffn = ffn @ linear2_weight.t() + linear2_bias  # [B*S, D]
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
    p = 16  # patch_size
    num_patches_h = H // p
    num_patches_w = W // p

    # Patch embedding: unfold + reshape + linear
    x = img.unfold(2, p, p).unfold(3, p, p)  # [B, C, nh, nw, p, p]
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, nh, nw, C, p, p]
    x = x.reshape(B, num_patches_h * num_patches_w, C * p * p)  # [B, 196, 768]

    # Linear projection
    x_2d = x.reshape(B * num_patches_h * num_patches_w, C * p * p)
    x_2d = x_2d @ p_patch_to_embedding_weight.t() + p_patch_to_embedding_bias
    x = x_2d.reshape(B, num_patches_h * num_patches_w, 512)  # [B, 196, 512]

    # Cat cls token and add positional embedding using Triton kernel
    seq_len = num_patches_h * num_patches_w + 1  # 197
    dim = 512
    out = torch.empty(B, seq_len, dim, device=img.device, dtype=img.dtype)
    N = B * seq_len * dim
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    patch_embed_cat_add_kernel[grid](
        p_cls_token.reshape(-1),  # [512]
        x,
        p_pos_embedding.reshape(-1),  # [197*512]
        out,
        B, seq_len, dim, N,
    )
    x = out

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
         p_transformer_layers_4_norm2_weight, p_transformer_layers_4_norm2_bias),
        (p_transformer_layers_5_self_attn_in_proj_weight, p_transformer_layers_5_self_attn_in_proj_bias,
         p_transformer_layers_5_self_attn_out_proj_weight, p_transformer_layers_5_self_attn_out_proj_bias,
         p_transformer_layers_5_norm1_weight, p_transformer_layers_5_norm1_bias,
         p_transformer_layers_5_linear1_weight, p_transformer_layers_5_linear1_bias,
         p_transformer_layers_5_linear2_weight, p_transformer_layers_5_linear2_bias,
         p_transformer_layers_5_norm2_weight, p_transformer_layers_5_norm2_bias),
    ]

    for params in layer_params:
        (in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
         norm1_weight, norm1_bias, linear1_weight, linear1_bias,
         linear2_weight, linear2_bias, norm2_weight, norm2_bias) = params
        x = transformer_layer(
            x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
            norm1_weight, norm1_bias, linear1_weight, linear1_bias,
            linear2_weight, linear2_bias, norm2_weight, norm2_bias,
        )

    # CLS token
    cls = x[:, 0]  # [B, 512]

    # MLP head: Linear + GELU + Dropout + Linear
    h = cls @ p_mlp_head_0_weight.t() + p_mlp_head_0_bias  # [B, 2048]
    h = triton_gelu(h)
    out = h @ p_mlp_head_3_weight.t() + p_mlp_head_3_bias  # [B, 10]

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

    p_patch_to_embedding_weight = torch.randn(dim, channels * patch_size ** 2)
    p_patch_to_embedding_bias = torch.randn(dim)
    p_cls_token = torch.randn(1, 1, dim)
    num_patches = (image_size // patch_size) ** 2
    p_pos_embedding = torch.randn(1, num_patches + 1, dim)

    layer_params = []
    for _ in range(depth):
        layer_params.extend([
            torch.randn(3 * dim),       # in_proj_bias
            torch.randn(3 * dim, dim),  # in_proj_weight
            torch.randn(dim, dim),      # out_proj_weight
            torch.randn(dim),           # out_proj_bias
            torch.randn(dim),           # norm1_weight
            torch.randn(dim),           # norm1_bias
            torch.randn(mlp_dim, dim),  # linear1_weight
            torch.randn(mlp_dim),       # linear1_bias
            torch.randn(dim, mlp_dim),  # linear2_weight
            torch.randn(dim),           # linear2_bias
            torch.randn(dim),           # norm2_weight
            torch.randn(dim),           # norm2_bias
        ])

    p_mlp_head_0_weight = torch.randn(mlp_dim, dim)
    p_mlp_head_0_bias = torch.randn(mlp_dim)
    p_mlp_head_3_weight = torch.randn(num_classes, mlp_dim)
    p_mlp_head_3_bias = torch.randn(num_classes)

    img = torch.randn(2, channels, image_size, image_size)

    return [
        p_patch_to_embedding_weight,
        p_patch_to_embedding_bias,
        p_cls_token,
        p_pos_embedding,
        *layer_params,
        p_mlp_head_0_weight,
        p_mlp_head_0_bias,
        p_mlp_head_3_weight,
        p_mlp_head_3_bias,
        img,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    # Move to CUDA
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                image_size = 224
                patch_size = 16
                num_classes = 10
                dim = 512
                depth = 6
                heads = 8
                mlp_dim = 2048
                channels = 3
                dropout = 0.0
                emb_dropout = 0.0

                num_patches = (image_size // patch_size) ** 2
                patch_dim = channels * patch_size ** 2

                self.patch_size = patch_size
                self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
                self.patch_to_embedding = nn.Linear(patch_dim, dim)
                self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
                self.dropout = nn.Dropout(emb_dropout)

                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True),
                    num_layers=depth
                )

                self.to_cls_token = nn.Identity()
                self.mlp_head = nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, num_classes)
                )

            def forward(self, img):
                p = self.patch_size
                x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0],