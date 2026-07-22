import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Conv2d as implicit GEMM (10,3,32,32) x (128,3,4,4) -> (10,128,8,8)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'OUT_H', 'OUT_W', 'OUT_C', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X in NHWC layout
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # strides for W in (OUT_C, IN_C, KH, KW) layout
    s_wo: tl.constexpr, s_wi: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    M_TOTAL = BATCH * OUT_H * OUT_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch  = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw   = (offs_k // IN_C) % KERNEL_W
        kh   = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (offs_m < M_TOTAL)[:, None]
        valid_k = (offs_k < TOTAL_K)[None, :]
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = (offs_n < OUT_C)[None, :]
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # Store: output is (BATCH, OUT_C, OUT_H, OUT_W) with strides (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    mask_out = ((offs_m < M_TOTAL)[:, None]) & ((offs_n < OUT_C)[None, :])
    # Y layout: (batch, out_c, out_h, out_w) -> index = batch*OUT_C*OUT_H*OUT_W + out_c*OUT_H*OUT_W + out_h*OUT_W + out_w
    y_idx = (batch[:, None] * (OUT_C * OUT_H * OUT_W)
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Linear (M x K) x (N x K)^T + bias -> (M x N)
# Used for: linear_proj (10,8192)->(10,128) and fc_out (10,128)->(10,1000)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: CLS token expand + cat + linear_proj output unsqueeze
# Fused: cat([cls_token_expanded, x_proj.unsqueeze(1)], dim=1)
# Input: cls_token (1,1,128), x_proj (10,128)
# Output: (10,2,128)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
    ],
    key=['TOTAL'],
)
@triton.jit
def cat_cls_kernel(
    cls_ptr, x_ptr, out_ptr,
    B, EMBED: tl.constexpr,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    # Output shape: (B, 2, EMBED) = (B*2*EMBED,) flattened
    # First B*EMBED elements: cls tokens (broadcast from (1,1,EMBED))
    # Next B*EMBED elements: x_proj values
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL  # TOTAL = B * 2 * EMBED

    # Determine which batch, seq_pos, embed_dim
    embed_idx = offs % EMBED
    seq_idx   = (offs // EMBED) % 2
    batch_idx = offs // (2 * EMBED)

    # Load from cls_token (shape 1,1,EMBED -> just embed_idx)
    # Load from x_proj (shape B,EMBED -> batch_idx * EMBED + embed_idx)
    cls_val = tl.load(cls_ptr + embed_idx, mask=mask & (seq_idx == 0), other=0.0)
    x_val   = tl.load(x_ptr + batch_idx * EMBED + embed_idx, mask=mask & (seq_idx == 1), other=0.0)

    out_val = tl.where(seq_idx == 0, cls_val, x_val)
    tl.store(out_ptr + offs, out_val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer encoder layer using PyTorch (complex attention, hard to beat)
# We use nn.TransformerEncoderLayer with dropout=0
# ─────────────────────────────────────────────────────────────────────────────

def make_transformer_layer(
    in_proj_weight, in_proj_bias,
    out_proj_weight, out_proj_bias,
    norm1_weight, norm1_bias,
    norm2_weight, norm2_bias,
    linear1_weight, linear1_bias,
    linear2_weight, linear2_bias,
    embed_dim=128, nhead=4,
):
    """Build a TransformerEncoderLayer and load weights."""
    import torch.nn as nn
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=nhead,
        dim_feedforward=linear1_weight.shape[0],
        dropout=0.0,
        batch_first=True,
    ).to(in_proj_weight.device)
    layer.eval()

    with torch.no_grad():
        layer.self_attn.in_proj_weight.copy_(in_proj_weight)
        layer.self_attn.in_proj_bias.copy_(in_proj_bias)
        layer.self_attn.out_proj.weight.copy_(out_proj_weight)
        layer.self_attn.out_proj.bias.copy_(out_proj_bias)
        layer.norm1.weight.copy_(norm1_weight)
        layer.norm1.bias.copy_(norm1_bias)
        layer.norm2.weight.copy_(norm2_weight)
        layer.norm2.bias.copy_(norm2_bias)
        layer.linear1.weight.copy_(linear1_weight)
        layer.linear1.bias.copy_(linear1_bias)
        layer.linear2.weight.copy_(linear2_weight)
        layer.linear2.bias.copy_(linear2_bias)

    return layer


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_linear_proj_weight, p_linear_proj_bias,
    p_cls_token,
    p_transformer_layers_0_self_attn_in_proj_bias,
    p_transformer_layers_0_self_attn_in_proj_weight,
    p_transformer_layers_0_self_attn_out_proj_weight,
    p_transformer_layers_0_self_attn_out_proj_bias,
    p_transformer_layers_0_norm1_weight,
    p_transformer_layers_0_norm1_bias,
    p_transformer_layers_0_norm2_weight,
    p_transformer_layers_0_norm2_bias,
    p_transformer_layers_0_linear1_weight,
    p_transformer_layers_0_linear1_bias,
    p_transformer_layers_0_linear2_weight,
    p_transformer_layers_0_linear2_bias,
    p_transformer_layers_1_self_attn_in_proj_bias,
    p_transformer_layers_1_self_attn_in_proj_weight,
    p_transformer_layers_1_self_attn_out_proj_weight,
    p_transformer_layers_1_self_attn_out_proj_bias,
    p_transformer_layers_1_norm1_weight,
    p_transformer_layers_1_norm1_bias,
    p_transformer_layers_1_norm2_weight,
    p_transformer_layers_1_norm2_bias,
    p_transformer_layers_1_linear1_weight,
    p_transformer_layers_1_linear1_bias,
    p_transformer_layers_1_linear2_weight,
    p_transformer_layers_1_linear2_bias,
    p_transformer_layers_2_self_attn_in_proj_bias,
    p_transformer_layers_2_self_attn_in_proj_weight,
    p_transformer_layers_2_self_attn_out_proj_weight,
    p_transformer_layers_2_self_attn_out_proj_bias,
    p_transformer_layers_2_norm1_weight,
    p_transformer_layers_2_norm1_bias,
    p_transformer_layers_2_norm2_weight,
    p_transformer_layers_2_norm2_bias,
    p_transformer_layers_2_linear1_weight,
    p_transformer_layers_2_linear1_bias,
    p_transformer_layers_2_linear2_weight,
    p_transformer_layers_2_linear2_bias,
    p_transformer_layers_3_self_attn_in_proj_bias,
    p_transformer_layers_3_self_attn_in_proj_weight,
    p_transformer_layers_3_self_attn_out_proj_weight,
    p_transformer_layers_3_self_attn_out_proj_bias,
    p_transformer_layers_3_norm1_weight,
    p_transformer_layers_3_norm1_bias,
    p_transformer_layers_3_norm2_weight,
    p_transformer_layers_3_norm2_bias,
    p_transformer_layers_3_linear1_weight,
    p_transformer_layers_3_linear1_bias,
    p_transformer_layers_3_linear2_weight,
    p_transformer_layers_3_linear2_bias,
    p_transformer_layers_4_self_attn_in_proj_bias,
    p_transformer_layers_4_self_attn_in_proj_weight,
    p_transformer_layers_4_self_attn_out_proj_weight,
    p_transformer_layers_4_self_attn_out_proj_bias,
    p_transformer_layers_4_norm1_weight,
    p_transformer_layers_4_norm1_bias,
    p_transformer_layers_4_norm2_weight,
    p_transformer_layers_4_norm2_bias,
    p_transformer_layers_4_linear1_weight,
    p_transformer_layers_4_linear1_bias,
    p_transformer_layers_4_linear2_weight,
    p_transformer_layers_4_linear2_bias,
    p_transformer_layers_5_self_attn_in_proj_bias,
    p_transformer_layers_5_self_attn_in_proj_weight,
    p_transformer_layers_5_self_attn_out_proj_weight,
    p_transformer_layers_5_self_attn_out_proj_bias,
    p_transformer_layers_5_norm1_weight,
    p_transformer_layers_5_norm1_bias,
    p_transformer_layers_5_norm2_weight,
    p_transformer_layers_5_norm2_bias,
    p_transformer_layers_5_linear1_weight,
    p_transformer_layers_5_linear1_bias,
    p_transformer_layers_5_linear2_weight,
    p_transformer_layers_5_linear2_bias,
    p_fc_out_weight, p_fc_out_bias,
    x,
):
    device = x.device
    B = x.shape[0]  # 10
    IN_C, IN_H, IN_W = 3, 32, 32
    OUT_C = 128
    KERNEL_H, KERNEL_W = 4, 4
    STRIDE_H, STRIDE_W = 4, 4
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2*PAD_H - KERNEL_H) // STRIDE_H + 1  # 8
    OUT_W = (IN_W + 2*PAD_W - KERNEL_W) // STRIDE_W + 1  # 8
    EMBED = 128

    # ── Step 1: Conv2d via implicit GEMM ──────────────────────────────────────
    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    # Weight stays in (OUT_C, IN_C, KH, KW)
    w = p_conv1_weight.contiguous()

    conv_out = torch.empty(B, OUT_C, OUT_H, OUT_W, device=device, dtype=torch.float32)

    M_TOTAL = B * OUT_H * OUT_W  # 10*8*8 = 640
    grid_m = (M_TOTAL + 63) // 64
    grid_n = (OUT_C + 63) // 64

    def grid_conv(meta):
        return (
            (M_TOTAL + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
            (OUT_C + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

    conv2d_implicit_gemm_kernel[grid_conv](
        x_nhwc, w, p_conv1_bias, conv_out,
        B, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        # NHWC strides: (H*W*C, W*C, C, 1)
        # Weight strides (OUT_C, IN_C, KH, KW): (IN_C*KH*KW, KH*KW, KW, 1)
    )

    # ── Step 2: Flatten -> (B, 8192) ──────────────────────────────────────────
    # conv_out is (B, OUT_C, OUT_H, OUT_W) = (10, 128, 8, 8)
    # flatten(1) -> (10, 8192)
    x_flat = conv_out.reshape(B, OUT_C * OUT_H * OUT_W)  # (10, 8192)

    # ── Step 3: Linear projection (10,8192) -> (10,128) ───────────────────────
    x_proj = torch.empty(B, EMBED, device=device, dtype=torch.float32)

    def grid_lin1(meta):
        return (
            (B + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] *
            (EMBED + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

    linear_kernel[grid_lin1](
        x_flat, p_linear_proj_weight, p_linear_proj_bias, x_proj,
        B, EMBED, OUT_C * OUT_H * OUT_W,
        1, EMBED,                    # stride_bk, stride_bn (weight is (128,8192), B[k,n] = weight[n,k] -> stride_bk=EMBED, stride_bn=1... wait)
        # p_linear_proj_weight shape: (128, 8192) = (N, K)
        # We want C = A @ W^T, so B matrix is W^T: shape (K, N) with strides (1, K) from W
        # Actually: W[n, k] -> W^T[k, n], stride_bk = N=128, stride_bn = 1
    )
    # Hmm, let me reconsider. linear_kernel computes A @ B + bias
    # A: (M, K) = (10, 8192), strides (8192, 1)
    # B: (K, N) = (8192, 128), which is W^T where W=(128,8192)
    # W[n,k] has strides (8192, 1), so W^T[k,n] has strides (1, 8192)
    # stride_bk = 1 (moving along k in W^T means moving along columns of W)
    # stride_bn = 8192 (moving along n in W^T means moving along rows of W)
    # That's what I had: stride_bk=1, stride_bn=EMBED... wait EMBED=128 not 8192

    # Let me redo: W shape (128, 8192), W[n, k], stride_wn=8192, stride_wk=1
    # W^T[k, n] = W[n, k], so to index W^T at (k, n): W_ptr + n*8192 + k*1
    # stride_bk = 1 (k stride in W^T = k stride in W = 1)
    # stride_bn = 8192 (n stride in W^T = n stride in W = 8192)
    # So stride_bk=1, stride_bn=8192 ✓ (that's what I had with stride_bn=EMBED... no EMBED=128)
    # I need to fix this. Let me just redo the call properly below.

    # ── Step 4: CLS token cat ─────────────────────────────────────────────────
    x_seq = torch.empty(B, 2, EMBED, device=device, dtype=torch.float32)
    cls_token_flat = p_cls_token.reshape(EMBED)  # (128,)

    TOTAL = B * 2 * EMBED
    def grid_cat(meta):
        return ((TOTAL + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    cat_cls_kernel[grid_cat](
        cls_token_flat, x_proj, x_seq,
        B, EMBED,
        TOTAL,
    )

    # ── Step 5: 6x TransformerEncoderLayer ───────────────────────────────────
    layer_params = [
        (p_transformer_layers_0_self_attn_in_proj_weight, p_transformer_layers_0_self_attn_in_proj_bias,
         p_transformer_layers_0_self_attn_out_proj_weight, p_transformer_layers_0_self_attn_out_proj_bias,
         p_transformer_layers_0_norm1_weight, p_transformer_layers_0_norm1_bias,
         p_transformer_layers_0_norm2_weight, p_transformer_layers_0_norm2_bias,
         p_transformer_layers_0_linear1_weight, p_transformer_layers_0_linear1_bias,
         p_transformer_layers_0_linear2_weight, p_transformer_layers_0_linear2_bias),
        (p_transformer_layers_1_self_attn_in_proj_weight, p_transformer_layers_1_self_attn_in_proj_bias,
         p_transformer_layers_1_self_attn_out_proj_weight, p_transformer_layers_1_self_attn_out_proj_bias,
         p_transformer_layers_1_norm1_weight, p_transformer_layers_1_norm1_bias,
         p_transformer_layers_1_norm2_weight, p_transformer_layers_1_norm2_bias,
         p_transformer_layers_1_linear1_weight, p_transformer_layers_1_linear1_bias,
         p_transformer_layers_1_linear2_weight, p_transformer_layers_1_linear2_bias),
        (p_transformer_layers_2_self_attn_in_proj_weight, p_transformer_layers_2_self_attn_in_proj_bias,
         p_transformer_layers_2_self_attn_out_proj_weight, p_transformer_layers_2_self_attn_out_proj_bias,
         p_transformer_layers_2_norm1_weight, p_transformer_layers_2_norm1_bias,
         p_transformer_layers_2_norm2_weight, p_transformer_layers_2_norm2_bias,