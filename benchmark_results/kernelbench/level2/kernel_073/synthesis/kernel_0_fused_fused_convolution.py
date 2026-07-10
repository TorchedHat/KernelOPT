import torch
import triton
import triton.language as tl
import math

# Convolution parameters
BATCH = 128
IN_C = 8
IN_H = 128
IN_W = 128
OUT_C = 64
OUT_H = 126
OUT_W = 126
KERNEL_H = 3
KERNEL_W = 3
STRIDE_H = 1
STRIDE_W = 1
PAD_H = 0
PAD_W = 0
SCALING_FACTOR = 2.0
BN_EPS = 1e-5

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 8},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M_TOTAL', 'OUT_C_ARG', 'TOTAL_K'],
)
@triton.jit
def conv_bn_scale_kernel(
    X_ptr, W_ptr, bias_ptr,
    bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    Y_ptr,
    # Dimensions
    M_TOTAL: tl.constexpr,
    OUT_C_ARG: tl.constexpr,
    TOTAL_K: tl.constexpr,
    # Spatial
    BATCH_ARG: tl.constexpr,
    IN_C_ARG: tl.constexpr,
    IN_H_ARG: tl.constexpr,
    IN_W_ARG: tl.constexpr,
    OUT_H_ARG: tl.constexpr,
    OUT_W_ARG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    # Strides for X (NHWC layout)
    s_xn: tl.constexpr,
    s_xh: tl.constexpr,
    s_xw: tl.constexpr,
    # s_xc = 1 (NHWC)
    # Strides for W (OUT_C, IN_C, KH, KW) -> flattened as (OUT_C, IN_C*KH*KW)
    # W stored as (OUT_C, KH, KW, IN_C) for coalesced access - we'll handle indexing
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W_ARG
    tmp = offs_m // OUT_W_ARG
    out_h_idx = tmp % OUT_H_ARG
    batch_idx = tmp // OUT_H_ARG

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer for X: batch offset in NHWC
    # X is NHWC: stride_n = IN_H*IN_W*IN_C, stride_h = IN_W*IN_C, stride_w = IN_C, stride_c = 1
    x_batch_offset = batch_idx * s_xn  # shape [BLOCK_M]

    # W is stored as (OUT_C, IN_C, KH, KW) originally
    # We'll iterate over K = IN_C * KH * KW
    # For each k: ic = k % IN_C, kw_i = (k // IN_C) % KW, kh_i = k // (IN_C * KW)
    # But we need W in a layout that allows tl.dot
    # W_ptr points to weight in (OUT_C, IN_C*KH*KW) layout (pre-transposed)
    # Actually we need [BLOCK_K, BLOCK_N] for tl.dot(x[BLOCK_M,BLOCK_K], w[BLOCK_K,BLOCK_N])

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode k -> (kh_i, kw_i, ic)
        ic = offs_k % IN_C_ARG
        kw_i = (offs_k // IN_C_ARG) % KW
        kh_i = offs_k // (IN_C_ARG * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * SH + kh_i[None, :] - PH
        iw = out_w_idx[:, None] * SW + kw_i[None, :] - PW

        # Validity mask
        valid_m = (offs_m < M_TOTAL)[:, None]
        valid_k = (offs_k < TOTAL_K)[None, :]
        valid_spatial = (ih >= 0) & (ih < IN_H_ARG) & (iw >= 0) & (iw < IN_W_ARG)
        mask_x = valid_m & valid_k & valid_spatial

        # X pointers: NHWC layout
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W: stored as (OUT_C, IN_C*KH*KW) but we need (BLOCK_K, BLOCK_N)
        # W original: (OUT_C, IN_C, KH, KW) with strides (IN_C*KH*KW, KH*KW, KW, 1)
        # Flattened index for W: out_c * (IN_C*KH*KW) + kh_i * (IN_C*KW) + kw_i * IN_C + ic
        # But we want W[k, n] = W[ic + kw_i*IN_C + kh_i*IN_C*KW, n]
        # W_ptr is (OUT_C, IN_C*KH*KW) -> W[n, k] so we need W[offs_n, offs_k]
        # w_ptrs shape: [BLOCK_K, BLOCK_N]
        w_flat_k = ic + kw_i * IN_C_ARG + kh_i * (IN_C_ARG * KW)  # [BLOCK_K]
        valid_wk = (offs_k < TOTAL_K)[:, None]
        valid_wn = (offs_n < OUT_C_ARG)[None, :]
        mask_w = valid_wk & valid_wn
        w_ptrs = W_ptr + w_flat_k[:, None] + offs_n[None, :] * TOTAL_K
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Now apply: bias + BN + scale
    mask_out_m = offs_m < M_TOTAL
    mask_out_n = offs_n < OUT_C_ARG
    mask_out = mask_out_m[:, None] & mask_out_n[None, :]

    # Load per-channel parameters
    bias_val = tl.load(bias_ptr + offs_n, mask=mask_out_n, other=0.0)
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=mask_out_n, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_n, mask=mask_out_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=mask_out_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=mask_out_n, other=0.0)

    # Add conv bias
    acc = acc + bias_val[None, :]

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(bn_var + EPS)
    acc = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # Scale
    acc = acc * SCALE

    # Store: output is (BATCH, OUT_C, OUT_H, OUT_W) in NCHW
    # Linear index: batch * OUT_C*OUT_H*OUT_W + out_c * OUT_H*OUT_W + out_h * OUT_W + out_w
    out_idx = (batch_idx[:, None] * (OUT_C_ARG * OUT_H_ARG * OUT_W_ARG)

    tl.store(Y_ptr + out_idx, acc.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, p_bn_weight, p_bn_bias,
    # x: (128, 8, 128, 128) NCHW
    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (128, 128, 128, 8)

    # Reshape weight to (OUT_C, IN_C*KH*KW) for easy indexing
    # Original: (64, 8, 3, 3) -> (64, 72)
    w_flat = p_conv_weight.reshape(OUT_C, IN_C * KERNEL_H * KERNEL_W).contiguous()
    # We need W[k, n] layout: (72, 64)
    w_kn = w_flat.t().contiguous()  # (72, 64)

    N, H, W_in, C = x_nhwc.shape
    OUT_H_val = (H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W_val = (W_in + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1
    M_TOTAL_val = N * OUT_H_val * OUT_W_val
    TOTAL_K_val = IN_C * KERNEL_H * KERNEL_W

    # Output: (BATCH, OUT_C, OUT_H, OUT_W)
    y = torch.empty(BATCH, OUT_C, OUT_H_val, OUT_W_val, device=x.device, dtype=x.dtype)

    # NHWC strides
    s_xn = IN_H * IN_W * IN_C  # 128*128*8 = 131072
    s_xh = IN_W * IN_C          # 128*8 = 1024
    s_xw = IN_C                  # 8

    BLOCK_M_val = 256
    BLOCK_N_val = 64

    grid = lambda meta: (
        triton.cdiv(M_TOTAL_val, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv_bn_scale_kernel[grid](
        x_nhwc, w_kn, p_conv_bias,
        p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var,
        y,
        M_TOTAL_val, OUT_C, TOTAL_K_val,
        BATCH, IN_C, IN_H, IN_W,
        OUT_H_val, OUT_W_val,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw,
        SCALING_FACTOR, BN_EPS,
    )

    return y


def get_inputs():
    p_conv_weight = torch.randn(64, 8, 3, 3, dtype=torch.float32, device='cuda')
    p_conv_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    p_bn_weight = torch.randn(64, dtype=torch.float32, device='cuda')
    p_bn_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    b_bn_running_mean = torch.randn(64, dtype=torch.float32, device='cuda')
    b_bn_running_var = torch.abs(torch.randn(64, dtype=torch.float32, device='cuda')) + 0.1
    x = torch.randn(128, 8, 128, 128, dtype=torch.float32, device='cuda')
    return [p_conv_weight, p_conv_bias, p_bn_weight, p_bn_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        p_conv_weight, p_conv_bias, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, x = inputs

        # Reference computation
        conv = nn.Conv2d(IN_C, OUT_C, KERNEL_H, bias=True)
        conv.weight.data = p_conv_weight
        conv.bias.data = p_conv_bias
        conv = conv.cuda()

        bn = nn.BatchNorm2d(OUT_C)
        bn.weight.data = p_bn_weight
        bn.bias.data = p_bn_bias
        bn.running_mean.data = b_bn_running_mean
        bn.running_var.data = b_bn_running_var
        bn.eval()
        bn = bn.cuda()

        with torch.no_grad():
            ref = conv(x)
            ref = bn(ref)
            ref = ref * SCALING_FACTOR

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")