import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: Implicit GEMM Convolution ────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=2, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_nhwc_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # X strides (NHWC): n, h, w, c
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # W strides (out_c, in_c, kh, kw) stored as (out_c, kh, kw, in_c) NHWC-like
    s_woc: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr, s_wic: tl.constexpr,
    # Out strides (NHWC): n, h, w, c
    s_on: tl.constexpr, s_oh: tl.constexpr, s_ow: tl.constexpr, s_oc: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx   = (offs_k // IN_C) % KERNEL_W
        kh_idx   = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid = (valid_m[:, None] & valid_k[None, :] &
                 (ih >= 0) & (ih < IN_H) &
                 (iw >= 0) & (iw < IN_W))

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  in_c_idx[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: shape (OUT_C, IN_C, KH, KW) -> stored NHWC-like (OUT_C, KH, KW, IN_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_woc +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw +
                  in_c_idx[:, None] * s_wic)
        valid_w = (valid_k[:, None] & (offs_n[None, :] < OUT_C))
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store output (NHWC)
    valid_out = ((offs_m < M_TOTAL)[:, None] & (offs_n < OUT_C)[None, :])
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * s_on +
                out_h_idx[:, None] * s_oh +
                out_w_idx[:, None] * s_ow +
                offs_n[None, :] * s_oc)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


# ─── Kernel 2: Fused InstanceNorm + bias + div ───────────────────────────────
# Each program handles one (batch, channel) pair -> reduces over OUT_H * OUT_W
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def instance_norm_div_kernel(
    # Input: NHWC layout (batch, out_h, out_w, out_c)
    In_ptr,
    # Bias: shape (OUT_C,)
    Bias_ptr,
    # Output: NCHW layout (batch, out_c, out_h, out_w)
    Out_ptr,
    BATCH: tl.constexpr,
    OUT_C: tl.constexpr,
    # Input strides (NHWC)
    s_in_n: tl.constexpr,
    s_in_h: tl.constexpr,
    s_in_w: tl.constexpr,
    s_in_c: tl.constexpr,
    # Output strides (NCHW)
    s_out_n: tl.constexpr,
    s_out_c: tl.constexpr,
    s_out_h: tl.constexpr,
    s_out_w: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    DIVIDE_BY: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch_idx = pid // OUT_C
    c_idx = pid % OUT_C

    # Load bias for this channel
    bias_val = tl.load(Bias_ptr + c_idx).to(tl.float32)

    # Base pointer for this (batch, channel) in NHWC input
    # In NHWC: element at (n, h, w, c) = n*s_in_n + h*s_in_h + w*s_in_w + c*s_in_c
    # For fixed n=batch_idx, c=c_idx, we iterate over hw = h*OUT_W + w
    in_base = In_ptr + batch_idx * s_in_n + c_idx * s_in_c

    # Pass 1: compute mean and variance using Welford's algorithm
    mean = tl.zeros((), dtype=tl.float32)
    m2   = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    for hw_off in range(0, HW, BLOCK_N):
        hw_idx = hw_off + tl.arange(0, BLOCK_N)
        mask = hw_idx < HW
        h_idx = hw_idx // OUT_W
        w_idx = hw_idx % OUT_W
        ptrs = in_base + h_idx * s_in_h + w_idx * s_in_w
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) + bias_val

        # Welford online update
        block_count = tl.sum(mask.to(tl.float32), axis=0)
        block_mean  = tl.sum(tl.where(mask, vals, tl.zeros_like(vals)), axis=0) / tl.maximum(block_count, 1.0)
        block_m2    = tl.sum(tl.where(mask, (vals - block_mean) * (vals - block_mean), tl.zeros_like(vals)), axis=0)

        # Merge with running stats
        new_count = count + block_count
        delta = block_mean - mean
        mean  = tl.where(new_count > 0.0,
                         mean + delta * block_count / tl.maximum(new_count, 1.0),
        m2    = m2 + block_m2 + delta * delta * count * block_count / tl.maximum(new_count, 1.0)
        count = new_count

    var = m2 / tl.maximum(count, 1.0)
    inv_std = libdevice.rsqrt(var + EPS)

    # Base pointer for output (NCHW)
    out_base = Out_ptr + batch_idx * s_out_n + c_idx * s_out_c

    # Pass 2: normalize, divide, store
    for hw_off in range(0, HW, BLOCK_N):
        hw_idx = hw_off + tl.arange(0, BLOCK_N)
        mask = hw_idx < HW
        h_idx = hw_idx // OUT_W
        w_idx = hw_idx % OUT_W
        ptrs = in_base + h_idx * s_in_h + w_idx * s_in_w
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) + bias_val

        normalized = (vals - mean) * inv_std
        result = normalized * (1.0 / DIVIDE_BY)

        out_ptrs = out_base + h_idx * s_out_h + w_idx * s_out_w
        tl.store(out_ptrs, result.to(tl.float32), mask=mask)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 64, 128, 128) NCHW
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

    # Convert weight to (OUT_C, KH, KW, IN_C) for coalesced K-dim access
    w_nhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    # Intermediate conv output in NHWC
    conv_out = torch.empty((BATCH, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Launch conv kernel
    def grid_conv(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv2d_nhwc_kernel[grid_conv](
        x_nhwc, w_nhwc, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        # X strides (NHWC)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides (OUT_C, KH, KW, IN_C)
        w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
        # Out strides (NHWC)
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
    )

    # Output in NCHW
    output = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    HW = OUT_H * OUT_W
    total_programs = BATCH * OUT_C

    def grid_norm(meta):
        return (total_programs,)

    instance_norm_div_kernel[grid_norm](
        conv_out,
        p_conv_bias,
        output,
        BATCH, OUT_C, HW,
        # Input strides (NHWC)
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        # Output strides (NCHW)
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        OUT_H, OUT_W,
        2.0,   # DIVIDE_BY
    )

    return output


def get_inputs():
    p_conv_weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias   = torch.randn(128, device='cuda', dtype=torch.float32)
    x             = torch.randn(128, 64, 128, 128, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        import torch.nn as nn
        p_conv_weight, p_conv_bias, x = inputs

        model = nn.Sequential()
        conv = nn.Conv2d(64, 128, 3, bias=True)
        conv.weight.data.copy_(p_conv_weight)
        conv.bias.data.copy_(p_conv_bias)
        inst_norm = nn.InstanceNorm2d(128)

        with torch.no_grad():
            ref = conv(x)
            ref = inst_norm(ref)
            ref = ref / 2.0

        max_diff = (output - ref).abs().max().item()
        print(f"Max difference vs reference: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("PASSED: Results match within tolerance.")
        else:
            print("FAILED: Results differ!")