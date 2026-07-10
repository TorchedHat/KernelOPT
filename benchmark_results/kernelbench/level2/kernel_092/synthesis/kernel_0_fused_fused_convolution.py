import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: Implicit GEMM Conv2d ───────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (Co, Ci, R, S) -> stored as (Co, R, S, Ci) for coalescing
    s_wo, s_wh, s_ww, s_wi,
    # strides for Y (NCHW)
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_k = offs_k < TOTAL_K
        x_valid = (
            valid_m[:, None]
            & (ih >= 0) & (ih < IN_H)
            & (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=x_valid, other=0.0)

        # W layout: (Co, Ci, R, S) -> we access W[offs_n, in_c, kh, kw]
        w_ptrs = (
        )
        w_valid = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_valid, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + bias[None, :]

    # Store in NCHW layout: Y[batch, oc, oh, ow]
    y_ptrs = (
    )
    out_mask = valid_m[:, None] & valid_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 2: GroupNorm mean/var reduction ────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 256},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_R': 512},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_R': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_R': 2048}, num_stages=2, num_warps=16),
    ],
    key=['GROUP_SIZE', 'HW'],
)
@triton.jit
def groupnorm_mean_var_kernel(
    X_ptr, Mean_ptr, Var_ptr,
    N, G, C, HW,
    BLOCK_R: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    pid = tl.program_id(0)
    batch = pid // G
    group = pid % G

    # Elements in this group: GROUP_SIZE channels * HW spatial
    total = GROUP_SIZE * HW

    mean_acc = tl.zeros((), dtype=tl.float32)
    m2_acc = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    for off in range(0, total, BLOCK_R):
        offs = off + tl.arange(0, BLOCK_R)
        mask = offs < total

        # offs -> (local_c, hw_idx)
        hw_idx = offs % HW
        local_c = offs // HW
        global_c = group * GROUP_SIZE + local_c

        # X is NCHW: X[batch, global_c, h, w]
        # linear index: batch*C*HW + global_c*HW + hw_idx
        x_ptrs = X_ptr + batch * C * HW + global_c * HW + hw_idx
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Welford online update
        count_new = count + tl.sum(mask.to(tl.float32), axis=0)
        delta = x - mean_acc
        # Only update for valid elements
        valid_x = tl.where(mask, x, mean_acc)
        # Simple accumulation for mean
        mean_acc = mean_acc + tl.sum(tl.where(mask, delta / tl.maximum(count_new, 1.0), 0.0), axis=0)
        count = count_new
        m2_acc = m2_acc + tl.sum(tl.where(mask, (x - mean_acc) * delta, 0.0), axis=0)

    variance = m2_acc / tl.maximum(count, 1.0)

    tl.store(Mean_ptr + pid, mean_acc)
    tl.store(Var_ptr + pid, variance)


# ─── Kernel 3: Fused GroupNorm normalize + Tanh + HardSwish + Add + LogSumExp ──
# We do this in two sub-kernels:
#   3a: per spatial position, reduce over C channels for logsumexp
#   But GroupNorm normalize needs mean/var per group.
# 
# Strategy: one kernel per (batch, h, w) that:
#   - loads all 64 channels of conv output
#   - normalizes each channel using precomputed mean/var
#   - applies tanh, hardswish
#   - adds conv output
#   - computes logsumexp over 64 channels
#   - stores result

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 128}, num_stages=3, num_warps=8),
    ],
    key=['C', 'G'],
)
@triton.jit
def fused_norm_tanh_hswish_add_logsumexp_kernel(
    N, C, H, W,
    G,
    eps,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one (batch, h, w) position
    pid = tl.program_id(0)
    total_hw = H * W
    batch = pid // total_hw
    hw = pid % total_hw
    h_idx = hw // W
    w_idx = hw % W

    # Load all C channels for this spatial position
    offs_c = tl.arange(0, BLOCK_C)
    valid_c = offs_c < C

    # Conv output: X[batch, c, h, w] = Conv_ptr[batch*C*H*W + c*H*W + h*W + w]
    conv_ptrs = Conv_ptr + batch * C * H * W + offs_c * H * W + h_idx * W + w_idx
    x_conv = tl.load(conv_ptrs, mask=valid_c, other=0.0).to(tl.float32)

    # GroupNorm: for each channel c, group = c // GROUP_SIZE
    group_idx = offs_c // GROUP_SIZE
    mean = tl.load(Mean_ptr + batch * G + group_idx, mask=valid_c, other=0.0).to(tl.float32)
    var  = tl.load(Var_ptr  + batch * G + group_idx, mask=valid_c, other=0.0).to(tl.float32)

    x_norm = (x_conv - mean) * libdevice.rsqrt(var + eps)

    gn_w = tl.load(GN_W_ptr + offs_c, mask=valid_c, other=1.0).to(tl.float32)
    gn_b = tl.load(GN_B_ptr + offs_c, mask=valid_c, other=0.0).to(tl.float32)
    x_norm = x_norm * gn_w + gn_b

    # Tanh
    x_tanh = libdevice.tanh(x_norm)

    # HardSwish: x * relu6(x+3) / 6
    x_hs = x_tanh * tl.minimum(tl.maximum(x_tanh + 3.0, 0.0), 6.0) / 6.0

    # Residual add
    x_res = x_conv + x_hs

    # LogSumExp over C channels
    # max for numerical stability
    x_max = tl.max(tl.where(valid_c, x_res, float('-inf')), axis=0)
    x_exp = tl.exp(x_res - x_max)
    x_exp = tl.where(valid_c, x_exp, 0.0)
    x_sum = tl.sum(x_exp, axis=0)
    lse = x_max + tl.log(x_sum)

    # Store: Out[batch, 0, h, w]
    out_ptr = Out_ptr + batch * H * W + h_idx * W + w_idx
    tl.store(out_ptr, lse)


def kernel_function(p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, x):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    G = 16
    GROUP_SIZE = OUT_C // G
    eps = 1e-5

    # ── Step 1: Conv2d (implicit GEMM) ──────────────────────────────────────
    # Use NHWC layout for X for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    grid_conv = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid_conv](
        x_nhwc, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M_TOTAL,
        # X strides (NHWC)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides (Co, Ci, R, S)
        p_conv_weight.stride(0), p_conv_weight.stride(2), p_conv_weight.stride(3), p_conv_weight.stride(1),
        # Y strides (NCHW)
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
    )

    # ── Step 2: GroupNorm mean/var ───────────────────────────────────────────
    HW = OUT_H * OUT_W
    mean_buf = torch.empty((BATCH, G), device=x.device, dtype=torch.float32)
    var_buf  = torch.empty((BATCH, G), device=x.device, dtype=torch.float32)

    grid_gn = (BATCH * G,)
    groupnorm_mean_var_kernel[grid_gn](
        conv_out, mean_buf, var_buf,
        BATCH, G, OUT_C, HW,
        GROUP_SIZE,
    )

    # ── Step 3: Fused normalize + tanh + hardswish + add + logsumexp ─────────
    out = torch.empty((BATCH, 1, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    total_positions = BATCH * OUT_H * OUT_W
    grid_fused = (total_positions,)

    fused_norm_tanh_hswish_add_logsumexp_kernel[grid_fused](
        conv_out, mean_buf, var_buf,
        p_group_norm_weight, p_group_norm_bias,
        out,
        BATCH, OUT_C, OUT_H, OUT_W,
        G, GROUP_SIZE, eps,
    )

    return out


def get_inputs():
    torch.manual_seed(0)
    p_conv_weight = torch.randn(64, 8, 3, 3, device='cuda')
    p_conv_bias = torch.randn(64, device='cuda')
    p_group_norm_weight = torch.randn(64, device='cuda')
    p_group_norm_bias = torch.randn(64, device='cuda')
    x = torch.rand(128, 8, 128, 128, device='cuda')
    return [p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, x = inputs

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 64, 3)
                self.group_norm = nn.GroupNorm(16, 64, eps=1e-5)
                self.tanh = nn.Tanh()
                self.hard_swish = nn.Hardswish()

            def forward(self, x):
                x_conv = self.conv(x)
                x_norm = self.group_norm(x_conv)
                x_tanh = self.tanh(x_norm)
                x_hard_swish = self.hard_swish(x_tanh)
                x_res = x_conv + x_hard_swish
                x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
                return x_logsumexp

        model = Model().cuda()
        with torch.no_grad():
            model.conv.weight.data.copy_(p_conv_weight)
            model.conv.bias.data.copy_(p_conv_bias)
            model.group_norm.weight.data.copy_(p_group_norm_weight)
            model.group_norm.bias.data.copy_(p_group_norm_bias)
            ref = model(x)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")