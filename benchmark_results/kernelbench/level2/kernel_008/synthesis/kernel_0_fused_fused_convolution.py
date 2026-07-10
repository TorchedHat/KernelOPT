import torch
import triton
import triton.language as tl

# ─── Kernel 1: Conv3d (implicit GEMM) + bias add + div ───────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_div_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    # Input dims
    BATCH: tl.constexpr, IN_C: tl.constexpr,
    IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    # Output dims
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    # Kernel dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    # Strides for X (NCHW layout)
    sx_n: tl.constexpr, sx_c: tl.constexpr, sx_d: tl.constexpr, sx_h: tl.constexpr, sx_w: tl.constexpr,
    # Strides for W (out_c, in_c, kd, kh, kw)
    sw_oc: tl.constexpr, sw_ic: tl.constexpr, sw_kd: tl.constexpr, sw_kh: tl.constexpr, sw_kw: tl.constexpr,
    # Strides for Out
    so_n: tl.constexpr, so_c: tl.constexpr, so_d: tl.constexpr, so_h: tl.constexpr, so_w: tl.constexpr,
    # Total sizes
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    DIVISOR: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: M = BATCH * OUT_D * OUT_H * OUT_W
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IN_C * KD * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K: k = ic * KD*KH*KW + kd * KH*KW + kh * KW + kw
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        ic_idx = tmp_k // KD

        # Input coordinates
        id_ = out_d[:, None] + kd_idx[None, :]
        ih_ = out_h[:, None] + kh_idx[None, :]
        iw_ = out_w[:, None] + kw_idx[None, :]

        valid_m = (batch < BATCH) & (offs_m < M)
        valid_k = offs_k < K
        valid = valid_m[:, None] & valid_k[None, :] & \
                (id_ >= 0) & (id_ < IN_D) & \
                (ih_ >= 0) & (ih_ < IN_H) & \
                (iw_ >= 0) & (iw_ < IN_W)

        x_ptrs = (X_ptr +
                  batch[:, None] * sx_n +
                  ic_idx[None, :] * sx_c +
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: shape (OUT_C, IN_C, KD, KH, KW)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * sw_oc +
                  ic_idx[:, None] * sw_ic +
                  kd_idx[:, None] * sw_kd +
                  kh_idx[:, None] * sw_kh +
                  kw_idx[:, None] * sw_kw)
        valid_w = valid_k[:, None] & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Add bias and divide
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = (acc + bias[None, :]) * (1.0 / DIVISOR)

    # Store output
    valid_out = (batch < BATCH)[:, None] & \
                (out_d < OUT_D)[:, None] & \
                (out_h < OUT_H)[:, None] & \
                (out_w < OUT_W)[:, None] & \
                (offs_n < N)[None, :]

    out_ptrs = (Out_ptr +
                batch[:, None] * so_n +
                offs_n[None, :] * so_c +
                out_d[:, None] * so_d +
                out_h[:, None] * so_h +
                out_w[:, None] * so_w)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


# ─── Kernel 2: MaxPool3d(2,2,2) + AdaptiveAvgPool3d(1,1,1) + add bias + sum(dim=1) ───

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_N': 2}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_N': 4}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_N': 8}, num_warps=8, num_stages=1),
    ],
    key=['BATCH', 'OUT_C', 'PD', 'PH', 'PW'],
)
@triton.jit
def pool_bias_sum_kernel(
    In_ptr, Bias_ptr, Out_ptr,
    BATCH: tl.constexpr, OUT_C: tl.constexpr,
    # Input to this kernel: conv output after div, shape (BATCH, OUT_C, ID, IH, IW)
    # where ID=14, IH=62, IW=62
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    # MaxPool params: kernel=2, stride=2 -> output (PD, PH, PW) = (7, 31, 31)
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    # Strides for input
    si_n: tl.constexpr, si_c: tl.constexpr, si_d: tl.constexpr, si_h: tl.constexpr, si_w: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one (batch, channel_block) pair
    # Output: (BATCH, 1, 1, 1) after sum over channels
    pid = tl.program_id(0)
    batch = pid

    # We'll accumulate sum over all channels for this batch
    total = tl.zeros((1,), dtype=tl.float32)

    for c in range(OUT_C):
        # Compute global average pool over maxpool output
        # MaxPool: kernel=2, stride=2, no padding
        # For each maxpool output position (pd, ph, pw):
        #   max over (2d, 2h, 2w) window in conv output
        # Then average over all (PD*PH*PW) positions

        pool_sum = tl.zeros((1,), dtype=tl.float32)
        pool_count = PD * PH * PW

        for pd in range(PD):
            for ph in range(PH):
                for pw in range(PW):
                    # MaxPool 2x2x2 window
                    id0 = pd * 2
                    ih0 = ph * 2
                    iw0 = pw * 2

                    base = In_ptr + batch * si_n + c * si_c

                    v000 = tl.load(base + id0 * si_d + ih0 * si_h + iw0 * si_w)
                    v001 = tl.load(base + id0 * si_d + ih0 * si_h + (iw0 + 1) * si_w)
                    v010 = tl.load(base + id0 * si_d + (ih0 + 1) * si_h + iw0 * si_w)
                    v011 = tl.load(base + id0 * si_d + (ih0 + 1) * si_h + (iw0 + 1) * si_w)
                    v100 = tl.load(base + (id0 + 1) * si_d + ih0 * si_h + iw0 * si_w)
                    v101 = tl.load(base + (id0 + 1) * si_d + ih0 * si_h + (iw0 + 1) * si_w)
                    v110 = tl.load(base + (id0 + 1) * si_d + (ih0 + 1) * si_h + iw0 * si_w)
                    v111 = tl.load(base + (id0 + 1) * si_d + (ih0 + 1) * si_h + (iw0 + 1) * si_w)

                    mx = tl.maximum(tl.maximum(tl.maximum(v000, v001), tl.maximum(v010, v011)),
                                    tl.maximum(tl.maximum(v100, v101), tl.maximum(v110, v111)))
                    pool_sum += mx

        # Global avg pool: divide by pool_count
        avg = pool_sum / pool_count

        # Add bias
        bias_val = tl.load(Bias_ptr + c)
        avg += bias_val

        total += avg

    tl.store(Out_ptr + batch, total)


def kernel_function(p_conv_weight, p_conv_bias, p_bias, x):
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = p_conv_weight.shape

    OUT_D = IN_D - KD + 1  # 14
    OUT_H = IN_H - KH + 1  # 62
    OUT_W = IN_W - KW + 1  # 62

    DIVISOR = 2.0

    # Allocate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M = BATCH * OUT_D * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KD * KH * KW

    # Strides for x
    sx_n, sx_c, sx_d, sx_h, sx_w = x.stride()
    # Strides for weight
    sw_oc, sw_ic, sw_kd, sw_kh, sw_kw = p_conv_weight.stride()
    # Strides for output
    so_n, so_c, so_d, so_h, so_w = conv_out.stride()

    BLOCK_M = 256
    BLOCK_N = 16
    BLOCK_K = 16

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    conv3d_div_kernel[(grid_m, grid_n)](
        x, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        sx_n, sx_c, sx_d, sx_h, sx_w,
        sw_oc, sw_ic, sw_kd, sw_kh, sw_kw,
        so_n, so_c, so_d, so_h, so_w,
        M, N, K,
        DIVISOR,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # MaxPool output dims
    PD = OUT_D // 2  # 7
    PH = OUT_H // 2  # 31
    PW = OUT_W // 2  # 31

    # Output: (BATCH, 1, 1, 1)
    out = torch.empty((BATCH, 1, 1, 1), device=x.device, dtype=x.dtype)

    # Flatten bias to 1D
    bias_1d = p_bias.view(OUT_C)

    si_n, si_c, si_d, si_h, si_w = conv_out.stride()

    pool_bias_sum_kernel[(BATCH,)](
        conv_out, bias_1d, out.view(BATCH),
        BATCH, OUT_C,
        OUT_D, OUT_H, OUT_W,
        PD, PH, PW,
        si_n, si_c, si_d, si_h, si_w,
        BLOCK_N=1,
    )

    return out


def get_inputs():
    p_conv_weight = torch.randn(16, 8, 3, 3, 3)
    p_conv_bias = torch.randn(16)
    p_bias = torch.randn(16, 1, 1, 1)
    x = torch.rand(128, 8, 16, 64, 64)
    return [p_conv_weight, p_conv_bias, p_bias, x]


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
        # Verify against PyTorch reference
        import torch.nn as nn
        p_conv_weight, p_conv_bias, p_bias, x = inputs_cuda

        conv = nn.Conv3d(8, 16, 3, bias=True).cuda()
        with torch.no_grad():
            conv.weight.copy_(p_conv_weight)
            conv.bias.copy_(p_conv_bias)

        with torch.no_grad():
            ref = conv(x)
            ref = ref / 2.0
            ref = nn.MaxPool3d(2)(ref)
            ref = nn.AdaptiveAvgPool3d((1, 1, 1))(ref)
            ref = ref + p_bias
            ref = torch.sum(ref, dim=1, keepdim=True)

        print("Max diff:", (output - ref).abs().max().item())
        print("Kernel executed successfully.")