import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: squeeze conv (1x1, Cin=3->Cout=6) + ReLU
#   Treats input as (M=N*H*W, K=3) @ W^T(3,6) -> (M, 6) + bias + relu
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 8,  'BLOCK_K': 8},  num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def squeeze_conv1x1_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
        # Load X tile [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]  (W is stored as [N, K], so W[n,k])
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias and ReLU
    b_mask = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: expand1x1 conv (1x1, Cin=6->Cout=64) + ReLU
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def expand1x1_conv_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    b_mask = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: expand3x3 conv (3x3 pad=1, Cin=6->Cout=64) + ReLU
#   Implicit GEMM: M = N*H*W, K = Cin*KH*KW = 6*3*3 = 54, N_out = 64
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def expand3x3_conv_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch  = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, tl.cdiv(TOTAL_K, BLOCK_K)):
        offs_k = k_off * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_m[:, None] < M_TOTAL) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> treat as [OUT_C, TOTAL_K]
        # w[oc, k] where k = kh*KW*IN_C + kw*IN_C + ic  (same decode as above)
        # Actually weight layout: [OUT_C, IN_C, KH, KW]
        # w_ptr[oc, ic, kh, kw] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + kh*KW + kw
        w_k_idx = kh_idx * (KW * IN_C) + kw_idx * IN_C + in_c  # linear K index in weight
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * TOTAL_K + w_k_idx[:, None]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Bias + ReLU
    b_mask = offs_n < OUT_C
    bias = tl.load(B_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store: output layout [BATCH, OUT_C, OUT_H, OUT_W]
    # linear index: batch*(OUT_C*OUT_H*OUT_W) + oc*(OUT_H*OUT_W) + oh*OUT_W + ow
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    out_idx = (batch[:, None] * (OUT_C * OUT_H * OUT_W)
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: cat along dim=1 -- fused copy of expand1x1 and expand3x3 outputs
#   into a single [BATCH, 128, H, W] tensor
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def cat_relu_kernel(
    A_ptr, B_ptr, Out_ptr,
    BATCH, C_A, C_B, HW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Concatenate A[BATCH, C_A, H, W] and B[BATCH, C_B, H, W] along dim=1
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    C_total = C_A + C_B
    # Decode output index
    hw_idx = offs % HW
    c_idx  = (offs // HW) % C_total
    b_idx  = offs // (HW * C_total)

    # Determine source
    in_A = c_idx < C_A
    src_c_A = c_idx  # channel in A
    src_c_B = c_idx - C_A  # channel in B

    a_idx = b_idx * C_A * HW + src_c_A * HW + hw_idx
    b_idx2 = b_idx * C_B * HW + src_c_B * HW + hw_idx

    a_val = tl.load(A_ptr + a_idx, mask=mask & in_A, other=0.0)
    b_val = tl.load(B_ptr + b_idx2, mask=mask & ~in_A, other=0.0)

    out_val = tl.where(in_A, a_val, b_val)
    tl.store(Out_ptr + offs, out_val, mask=mask)


def kernel_function(
):
    BATCH, IN_C, H, W = x.shape
    SQ_C = p_squeeze_weight.shape[0]   # 6
    EX1_C = p_expand1x1_weight.shape[0]  # 64
    EX3_C = p_expand3x3_weight.shape[0]  # 64

    M = BATCH * H * W  # 128 * 256 * 256 = 8388608

    # ── Step 1: Squeeze conv 1x1 (3->6) + ReLU ──────────────────────────────
    # Input x: [BATCH, IN_C, H, W] -> treat as [M, IN_C] in NHWC-like view
    # For 1x1 conv, each spatial position is independent: x_flat[m, ic]
    # x is NCHW: x[b, c, h, w] = x_ptr + b*(C*H*W) + c*(H*W) + h*W + w
    # Reinterpret as [M, IN_C]: row m = (b, h, w), col = c
    # stride_m = 1 (adjacent pixels in same channel are stride 1 apart in HW)
    # Actually for NCHW: x[b,c,h,w] -> flat index b*C*H*W + c*H*W + h*W + w
    # For implicit GEMM with 1x1 conv, we need x as [M, K] where M=B*H*W, K=C
    # x_nhwc[m, c] = x[b, c, h, w] where m = b*H*W + h*W + w
    # stride_m in NCHW = 1 (w changes fastest), stride_c = H*W
    # So x_nhwc has stride_m=1, stride_c=H*W -- this is NOT contiguous in NHWC sense
    # We need to permute to NHWC for coalesced access

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [BATCH, H, W, IN_C]
    # Now x_nhwc[m, k] with stride_m=IN_C, stride_k=1

    sq_w = p_squeeze_weight.reshape(SQ_C, IN_C)  # [6, 3]
    # W stored as [N, K]: w[n, k] = sq_w[n, k], stride_n=IN_C, stride_k=1

    sq_out = torch.empty((M, SQ_C), device=x.device, dtype=torch.float32)

    grid_sq = (triton.cdiv(M, 128), triton.cdiv(SQ_C, 16))
    squeeze_conv1x1_relu_kernel[grid_sq](
        x_nhwc, sq_w, p_squeeze_bias, sq_out,
        M, SQ_C, IN_C,
    )
    # sq_out: [M, SQ_C] = [8388608, 6]

    # ── Step 2: Expand1x1 conv (6->64) + ReLU ───────────────────────────────
    ex1_w = p_expand1x1_weight.reshape(EX1_C, SQ_C)  # [64, 6]

    ex1_out = torch.empty((M, EX1_C), device=x.device, dtype=torch.float32)

    grid_ex1 = (triton.cdiv(M, 64), triton.cdiv(EX1_C, 64))
    expand1x1_conv_relu_kernel[grid_ex1](
        sq_out, ex1_w, p_expand1x1_bias, ex1_out,
        M, EX1_C, SQ_C,
    )
    # ex1_out: [M, EX1_C] = [8388608, 64]

    # ── Step 3: Expand3x3 conv (6->64, 3x3 pad=1) + ReLU ───────────────────
    # Input: sq_out [M, SQ_C] but we need it as [BATCH, SQ_C, H, W] for 3x3 conv
    # sq_out is [M, SQ_C] = [B*H*W, 6] in NHWC order
    # Reinterpret as [BATCH, H, W, SQ_C] then treat strides accordingly
    # For the 3x3 kernel, we need to access neighbors, so we need spatial structure

    # sq_out_nchw: [BATCH, SQ_C, H, W]
    sq_out_nchw = sq_out.reshape(BATCH, H, W, SQ_C).permute(0, 3, 1, 2).contiguous()

    ex3_w = p_expand3x3_weight.reshape(EX3_C, SQ_C * 3 * 3)  # [64, 54]

    ex3_out_flat = torch.empty((M, EX3_C), device=x.device, dtype=torch.float32)
    # We'll store as [BATCH, EX3_C, H, W] directly
    ex3_out = torch.empty((BATCH, EX3_C, H, W), device=x.device, dtype=torch.float32)

    TOTAL_K = SQ_C * 3 * 3  # 54
    OUT_H, OUT_W = H, W  # same size due to padding=1

    s_xn = SQ_C * H * W
    s_xc = H * W
    s_xh = W
    s_xw = 1

    grid_ex3 = (triton.cdiv(M, 64), triton.cdiv(EX3_C, 64))
    expand3x3_conv_relu_kernel[grid_ex3](
        sq_out_nchw, ex3_w, p_expand3x3_bias, ex3_out,
        BATCH, SQ_C, H, W,
        EX3_C, OUT_H, OUT_W,
        M, TOTAL_K,
        s_xn, s_xc, s_xh, s_xw,
    )

    # ── Step 4: Reshape ex1_out to NCHW and cat ─────────────────────────────
    # ex1_out is [M, EX1_C] in NHWC order -> reshape to [BATCH, H, W, EX1_C] -> permute to NCHW
    ex1_out_nchw = ex1_out.reshape(BATCH, H, W, EX1_C).permute(0, 3, 1, 2).contiguous()

    # Cat along dim=1: [BATCH, EX1_C+EX3_C, H, W]
    C_total = EX1_C + EX3_C
    out = torch.empty((BATCH, C_total, H, W), device=x.device, dtype=torch.float32)

    numel = BATCH * C_total * H * W
    grid_cat = (triton.cdiv(numel, 1024),)
    cat_relu_kernel[grid_cat](
        ex1_out_nchw, ex3_out, out,
        BATCH, EX1_C, EX3_C, H * W,
        numel,
    )

    return out


def get_inputs():
    batch_size = 128
    num_input_features = 3
    height, width = 256, 256
    squeeze_channels = 6
    expand1x1_channels = 64
    expand3x3_channels = 64

    p_squeeze_weight = torch.randn(squeeze_channels, num_input_features, 1, 1)
    p_squeeze_bias = torch.randn(squeeze_channels)
    p_expand1x1_weight = torch.randn(expand1x1_channels, squeeze_channels, 1, 1)
    p_expand1x1_bias = torch.randn(expand1x1_channels)
    p_expand3x3_weight = torch.randn(expand3x3_channels, squeeze_channels, 3, 3)
    p_expand3x3_bias = torch.randn(expand3x3_channels)
    x = torch.randn(batch_size, num_input_features, height, width)

    return [
        p_squeeze_weight.cuda(),
        p_squeeze_bias.cuda(),
        p_expand1x1_weight.cuda(),
        p_expand1x1_bias.cuda(),
        p_expand3x3_weight.cuda(),
        p_expand3x3_bias.cuda(),
        x.cuda(),
    ]


if __name__ == "__main__":
    import argparse
    import torch.nn as nn

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        p_squeeze_weight, p_squeeze_bias, p_expand1x1_weight, p_expand1x1_bias, \
            p_expand3x3_weight, p_expand3x3_bias, x = inputs

        import torch.nn.functional as F

        sq = F.conv2d(x, p_squeeze_weight, p_squeeze_bias)
        sq = F.relu(sq)
        ex1 = F.conv2d(sq, p_expand1x1_weight, p_expand1x1_bias)
        ex1 = F.relu(ex1)
        ex3 = F.conv2d(sq, p_expand3x3_weight, p_expand3x3_bias, padding=1)
        ex3 = F.relu(ex3)
        ref = torch.cat([ex1, ex3], dim=1)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")