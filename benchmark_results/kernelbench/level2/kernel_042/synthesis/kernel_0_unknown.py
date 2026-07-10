import torch
import triton
import triton.language as tl

# ConvTranspose2d kernel using implicit GEMM
# For ConvTranspose2d with stride=1, pad=0:
# out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh-kh, ow-kw] * w[ic, oc, kh, kw]
# where oh-kh in [0, IH), ow-kw in [0, IW)
# M = N * OH * OW, N_out = OC, K = IC * KH * KW

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    M_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    batch = offs_m // (OW * OH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IC * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (ic, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        ic = offs_k // (KW * KH)

        # For ConvTranspose2d: input position ih = oh - kh, iw = ow - kw
        ih = oh[:, None] - kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] - kw[None, :]  # [BLOCK_M, BLOCK_K]

        valid_x = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & \
                  (offs_k[None, :] < TOTAL_K) & (offs_m[:, None] < M_TOTAL)

        # X is in NCHW: x[n, ic, ih, iw] = X_ptr + n*IC*IH*IW + ic*IH*IW + ih*IW + iw
        x_ptrs = X_ptr + batch[:, None] * (IC * IH * IW) + ic[None, :] * (IH * IW) + ih * IW + iw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # W shape: [IC, OC, KH, KW] -> w[ic, oc, kh, kw]
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OC)
        w_ptrs = W_ptr + ic[:, None] * (OC * KH * KW) + offs_n[None, :] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias_vals = tl.load(B_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store output Y in NCHW: y[n, oc, oh, ow]
    valid_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OC)
    y_ptrs = Y_ptr + batch[:, None] * (OC * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


# Fused kernel: mean over spatial + add bias + logsumexp over channels + sum + mul
# Input Y: [N, OC, OH, OW]
# bias: [OC, 1, 1]
# Output: [N, 1]
# Steps:
#   1. mean_val[n, oc] = mean(Y[n, oc, :, :]) over OH*OW
#   2. z[n, oc] = mean_val[n, oc] + bias[oc]
#   3. lse[n] = log(sum_oc exp(z[n, oc]))  (logsumexp over oc)
#   4. result[n] = lse[n] * 1 (sum over 1x1 spatial = just lse[n])
#   5. result[n] = result[n] * 10.0

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 32}, num_warps=2, num_stages=2),
    ],
    key=['OC', 'SPATIAL'],
)
@triton.jit
def fused_reduction_kernel(
    Y_ptr, Bias_ptr, Out_ptr,
    N, OC, OH, OW,
    SPATIAL,
    BLOCK_C: tl.constexpr,
):
    # One program per batch element
    batch = tl.program_id(0)

    # Step 1 & 2: Compute mean over spatial + add bias for each channel
    # Then do logsumexp over channels
    # We need to do this in two passes over channels for numerical stability

    # Pass 1: find max over channels (for numerical stability of logsumexp)
    row_max = tl.full((), float('-inf'), dtype=tl.float32)

    for c_off in range(0, OC, BLOCK_C):
        offs_c = c_off + tl.arange(0, BLOCK_C)
        mask_c = offs_c < OC

        # Compute mean over spatial for this channel block
        # Y[batch, oc, :, :] -> sum over SPATIAL positions
        # We need to sum over OH*OW for each channel
        # Load spatial sum for each channel
        # This requires iterating over spatial... but SPATIAL=514*514=264196 is huge
        # We'll accumulate spatial sum inline

        # Actually, let's compute the spatial mean for each channel in this block
        # by iterating over spatial positions
        # But that would be O(OC * SPATIAL) per batch which is too slow for a single kernel

        # Better: load precomputed means (we'll compute them in a separate pass)
        # Actually let's just do it: for each channel in the block, sum over spatial

        # For now, use a simplified approach: load the mean directly
        # We need spatial_sum[batch, oc] = sum of Y[batch, oc, h, w] for all h,w
        spatial_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for sp_off in range(0, SPATIAL, 1024):
            sp_offs = sp_off + tl.arange(0, 1024)
            sp_mask = sp_offs < SPATIAL

            # Y[batch, oc, sp] in NCHW layout
            # y_ptr = Y_ptr + batch * OC * SPATIAL + oc * SPATIAL + sp
            y_ptrs = Y_ptr + batch * OC * SPATIAL + offs_c[:, None] * SPATIAL + sp_offs[None, :]
            y_vals = tl.load(y_ptrs, mask=mask_c[:, None] & sp_mask[None, :], other=0.0)
            spatial_sum += tl.sum(y_vals, axis=1)

        mean_vals = spatial_sum / SPATIAL

        bias_vals = tl.load(Bias_ptr + offs_c, mask=mask_c, other=0.0)
        z = mean_vals + bias_vals

        c_max = tl.max(tl.where(mask_c, z, tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)), axis=0)
        row_max = tl.maximum(row_max, c_max)

    # Pass 2: compute logsumexp
    row_sum = tl.zeros((), dtype=tl.float32)

    for c_off in range(0, OC, BLOCK_C):
        offs_c = c_off + tl.arange(0, BLOCK_C)
        mask_c = offs_c < OC

        spatial_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for sp_off in range(0, SPATIAL, 1024):
            sp_offs = sp_off + tl.arange(0, 1024)
            sp_mask = sp_offs < SPATIAL

            y_ptrs = Y_ptr + batch * OC * SPATIAL + offs_c[:, None] * SPATIAL + sp_offs[None, :]
            y_vals = tl.load(y_ptrs, mask=mask_c[:, None] & sp_mask[None, :], other=0.0)
            spatial_sum += tl.sum(y_vals, axis=1)

        mean_vals = spatial_sum / SPATIAL
        bias_vals = tl.load(Bias_ptr + offs_c, mask=mask_c, other=0.0)
        z = mean_vals + bias_vals

        exp_vals = tl.exp(tl.where(mask_c, z - row_max, tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)))
        row_sum += tl.sum(exp_vals, axis=0)

    lse = row_max + tl.log(row_sum)

    # logsumexp result is scalar per batch, sum over (1,1) spatial = itself, multiply by 10
    result = lse * 10.0

    tl.store(Out_ptr + batch, result)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x):
    # Weight shape for ConvTranspose2d: [IC, OC, KH, KW] = [64, 128, 3, 3]
    IC, OC, KH, KW = p_conv_transpose_weight.shape
    N, _, IH, IW = x.shape

    OH = IH + KH - 1  # 512 + 3 - 1 = 514
    OW = IW + KW - 1  # 512 + 3 - 1 = 514

    M_TOTAL = N * OH * OW

    # Allocate output for conv_transpose
    Y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    # Launch conv_transpose2d kernel
    def grid_conv(meta):
        return (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))

    conv_transpose2d_kernel[grid_conv](
        x, p_conv_transpose_weight, p_conv_transpose_bias, Y,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M_TOTAL,
    )

    # Allocate output for fused reduction
    SPATIAL = OH * OW
    Out = torch.empty((N, 1), device=x.device, dtype=x.dtype)

    # Launch fused reduction kernel (one program per batch)
    fused_reduction_kernel[(N,)](
        Y, p_bias.view(OC), Out.view(N),
        N, OC, OH, OW,
        SPATIAL,
    )

    return Out


def get_inputs():
    p_conv_transpose_weight = torch.randn(64, 128, 3, 3)
    p_conv_transpose_bias = torch.randn(128)
    p_bias = torch.randn(128, 1, 1)
    x = torch.rand(16, 64, 512, 512)
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x = inputs

        # Reference computation
        conv_t = nn.ConvTranspose2d(64, 128, 3, bias=True)
        with torch.no_grad():
            conv_t.weight.copy_(p_conv_transpose_weight)
            conv_t.bias.copy_(p_conv_transpose_bias)

        x_ref = x.cuda()
        with torch.no_grad():
            y_ref = conv_t.cuda()(x_ref)
            y_ref = torch.mean(y_ref, dim=(2, 3), keepdim=True)
            y_ref = y_ref + p_bias.cuda()
            y_ref = torch.logsumexp(y_ref, dim=1, keepdim=True)
            y_ref = torch.sum(y_ref, dim=(2, 3))
            y_ref = y_ref * 10.0

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {y_ref.shape}")
        print(f"Output: {output[:4]}")
        print(f"Ref: {y_ref[:4]}")
        max_diff = (output - y_ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")