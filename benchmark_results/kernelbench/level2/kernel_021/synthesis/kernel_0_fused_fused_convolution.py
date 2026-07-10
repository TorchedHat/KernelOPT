import torch
import triton
import triton.language as tl
import math

# Conv2d as implicit GEMM + fused epilogue (add, mul, sigmoid)
# Then a separate fused kernel for group norm (mean/var + normalize)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_sigmoid_kernel(
    X_ptr, W_ptr, bias_ptr, add_bias_ptr, scale_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K: layout is [KH, KW, IN_C] (NHWC-style weight)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        # X in NHWC: [N, H, W, C], stride: n*IN_H*IN_W*IN_C + h*IN_W*IN_C + w*IN_C + c
        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_H * IN_W * IN_C) +
            in_c[None, :]
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # W in [OUT_C, KH, KW, IN_C] -> NHWC weight: [OUT_C, KH*KW*IN_C]
        # We store W as [OUT_C, KH, KW, IN_C] flattened
        # w_ptrs: [BLOCK_K, BLOCK_N]
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + (
            offs_n[None, :] * (KH * KW * IN_C) +
            kh[:, None] * (KW * IN_C) +
            kw[:, None] * IN_C +
            in_c[:, None]
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add conv bias
    bias_mask = offs_n < OUT_C
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Add p_bias (shape [OUT_C, 1, 1])
    add_bias = tl.load(add_bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + add_bias[None, :]

    # Multiply scale (shape [OUT_C, 1, 1])
    scale = tl.load(scale_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc * scale[None, :]

    # Sigmoid
    acc = tl.sigmoid(acc)

    # Store output in NCHW: [BATCH, OUT_C, OUT_H, OUT_W]
    # index = batch * OUT_C*OUT_H*OUT_W + c * OUT_H*OUT_W + h * OUT_W + w
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    out_ptrs = Out_ptr + (
        batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# Group norm kernel: fused mean/var + normalize
# Input: [BATCH, OUT_C, OUT_H, OUT_W], groups=NUM_GROUPS
# Each program handles one (batch, group) pair
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
    ],
    key=['GROUP_SIZE', 'SPATIAL'],
)
@triton.jit
def group_norm_kernel(
    X_ptr, Out_ptr, GN_weight_ptr, GN_bias_ptr,
    BATCH: tl.constexpr, OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    NUM_GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr, SPATIAL: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program: one (batch, group) pair
    pid = tl.program_id(0)
    batch_idx = pid // NUM_GROUPS
    group_idx = pid % NUM_GROUPS

    # Elements per group: GROUP_SIZE * SPATIAL
    N = GROUP_SIZE * SPATIAL

    # Compute mean and variance using Welford's algorithm
    mean = tl.zeros((), dtype=tl.float32)
    m2 = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    base_offset = batch_idx * OUT_C * SPATIAL + group_idx * GROUP_SIZE * SPATIAL

    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # offs maps to: channel_in_group * SPATIAL + spatial_idx
        # In NCHW: channel = group_idx * GROUP_SIZE + offs // SPATIAL
        #          spatial = offs % SPATIAL
        ch_in_group = offs // SPATIAL
        sp = offs % SPATIAL
        ch = group_idx * GROUP_SIZE + ch_in_group
        x_ptrs = X_ptr + batch_idx * OUT_C * SPATIAL + ch * SPATIAL + sp
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Welford update
        count_new = count + tl.sum(mask.to(tl.float32), axis=0)
        delta = x - mean
        # Only update for valid elements
        valid_count = tl.sum(mask.to(tl.float32), axis=0)
        mean_new = mean + tl.sum(tl.where(mask, delta / tl.maximum(count_new, 1.0), 0.0), axis=0)
        delta2 = x - mean_new
        m2_new = m2 + tl.sum(tl.where(mask, delta * delta2, 0.0), axis=0)
        mean = mean_new
        m2 = m2_new
        count = count_new

    variance = m2 / tl.maximum(count, 1.0)
    inv_std = 1.0 / tl.sqrt(variance + eps)

    # Normalize
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        ch_in_group = offs // SPATIAL
        sp = offs % SPATIAL
        ch = group_idx * GROUP_SIZE + ch_in_group
        x_ptrs = X_ptr + batch_idx * OUT_C * SPATIAL + ch * SPATIAL + sp
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        x_norm = (x - mean) * inv_std

        # Apply weight and bias (per-channel)
        gn_w = tl.load(GN_weight_ptr + ch, mask=mask, other=1.0).to(tl.float32)
        gn_b = tl.load(GN_bias_ptr + ch, mask=mask, other=0.0).to(tl.float32)
        out = x_norm * gn_w + gn_b

        out_ptrs = Out_ptr + batch_idx * OUT_C * SPATIAL + ch * SPATIAL + sp
        tl.store(out_ptrs, out.to(tl.float32), mask=mask)


def kernel_function(
    p_conv_weight, p_conv_bias, p_bias, p_scale,
):
    BATCH = x.shape[0]
    IN_C = x.shape[1]
    IN_H = x.shape[2]
    IN_W = x.shape[3]
    OUT_C = p_conv_weight.shape[0]
    KH = p_conv_weight.shape[2]
    KW = p_conv_weight.shape[3]
    OUT_H = IN_H - KH + 1
    OUT_W = IN_W - KW + 1
    NUM_GROUPS = 8
    GROUP_SIZE = OUT_C // NUM_GROUPS
    SPATIAL = OUT_H * OUT_W

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Convert weight to [OUT_C, KH, KW, IN_C]
    w_nhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()

    # Flatten bias/scale
    add_bias_flat = p_bias.reshape(OUT_C).contiguous()
    scale_flat = p_scale.reshape(OUT_C).contiguous()

    # Output of conv+sigmoid: NCHW
    conv_out = torch.empty(BATCH, OUT_C, OUT_H, OUT_W, device=x.device, dtype=torch.float32)

    M_TOTAL = BATCH * OUT_H * OUT_W
    BLOCK_M = 128
    BLOCK_N = 32

    grid_conv = (
        triton.cdiv(M_TOTAL, BLOCK_M),
        triton.cdiv(OUT_C, BLOCK_N),
    )

    conv2d_sigmoid_kernel[grid_conv](
        x_nhwc, w_nhwc, p_conv_bias, add_bias_flat, scale_flat, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        BLOCK_M, BLOCK_N, 16,
    )

    # Group norm
    gn_out = torch.empty_like(conv_out)
    grid_gn = (BATCH * NUM_GROUPS,)

    group_norm_kernel[grid_gn](
        conv_out, gn_out, p_group_norm_weight, p_group_norm_bias,
        BATCH, OUT_C, OUT_H, OUT_W,
        NUM_GROUPS, GROUP_SIZE, SPATIAL,
        1e-5,
    )

    return gn_out


def get_inputs():
    p_conv_weight = torch.randn(32, 8, 3, 3, device='cuda')
    p_conv_bias = torch.randn(32, device='cuda')
    p_bias = torch.randn(32, 1, 1, device='cuda')
    p_scale = torch.randn(32, 1, 1, device='cuda')
    p_group_norm_weight = torch.randn(32, device='cuda')
    p_group_norm_bias = torch.randn(32, device='cuda')
    x = torch.rand(128, 8, 256, 256, device='cuda')
    return [p_conv_weight, p_conv_bias, p_bias, p_scale, p_group_norm_weight, p_group_norm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F

        p_conv_weight, p_conv_bias, p_bias, p_scale, p_group_norm_weight, p_group_norm_bias, x = inputs

        # Reference
        ref = F.conv2d(x, p_conv_weight, p_conv_bias)
        ref = ref + p_bias
        ref = ref * p_scale
        ref = torch.sigmoid(ref)
        ref = F.group_norm(ref, 8, p_group_norm_weight, p_group_norm_bias, eps=1e-5)

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.6e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")