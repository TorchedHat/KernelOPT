import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Output shape: [64, 32, 30, 62, 62]
# M = 64 * 30 * 62 * 62 = 7,372,800 output spatial positions
# N = 32 output channels
# K = 8 * 3 * 3 * 3 = 216 reduction dimension

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 8},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W', 'IN_C', 'KD', 'KH', 'KW'],
)
@triton.jit
def fused_conv3d_activations_kernel(
    X_ptr, W_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xd: tl.constexpr,
    stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_wco: tl.constexpr, stride_wci: tl.constexpr, stride_wkd: tl.constexpr,
    stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: batch * OUT_D * OUT_H * OUT_W
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    out_d_idx = tmp2 % OUT_D
    batch_idx = tmp2 // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KD * KH * KW
    BLOCK_K_COUNT = (IN_C + BLOCK_K - 1) // BLOCK_K

    for dijk in range(KD * KH * KW * BLOCK_K_COUNT):
        k_block = dijk % BLOCK_K_COUNT
        dij = dijk // BLOCK_K_COUNT
        kw = dij % KW
        di = dij // KW
        kh = di % KH
        kd = di // KH

        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        in_d = out_d_idx * 1 + kd
        in_h = out_h_idx * 1 + kh
        in_w = out_w_idx * 1 + kw

        x_ptrs = (X_ptr

        mask_x = ((batch_idx[:, None] < BATCH)
                  & (in_d[:, None] >= 0) & (in_d[:, None] < IN_D)
                  & (in_h[:, None] >= 0) & (in_h[:, None] < IN_H)
                  & (in_w[:, None] >= 0) & (in_w[:, None] < IN_W)

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = (W_ptr

        mask_w = (offs_k[:, None] < IN_C) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add conv bias
    conv_bias = tl.load(conv_bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + conv_bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # LeakyReLU (since input is already ReLU'd, all values >= 0, so leaky_relu is identity)
    # But we implement it correctly anyway
    acc = tl.where(acc > 0.0, acc, acc * 0.01)

    # GELU: 0.5 * x * (1 + erf(x * 0.7071067811865476))
    gelu_x = acc * 0.7071067811865476
    erf_val = libdevice.erf(gelu_x)
    acc = acc * 0.5 * (1.0 + erf_val)

    # Sigmoid
    acc = tl.sigmoid(acc)

    # Add bias (shape [32, 1, 1, 1] -> per output channel)
    bias_val = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias_val[None, :]

    # Store output
    mask_out = ((batch_idx[:, None] < BATCH)

    out_idx = (batch_idx[:, None] * (OUT_C * OUT_D * OUT_H * OUT_W)

    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, p_bias, x):
    # x: [64, 8, 32, 64, 64]
    # p_conv_weight: [32, 8, 3, 3, 3]
    # p_conv_bias: [32]
    # p_bias: [32, 1, 1, 1]

    BATCH  = x.shape[0]
    IN_C   = x.shape[1]
    IN_D   = x.shape[2]
    IN_H   = x.shape[3]
    IN_W   = x.shape[4]

    OUT_C  = p_conv_weight.shape[0]
    KD     = p_conv_weight.shape[2]
    KH     = p_conv_weight.shape[3]
    KW     = p_conv_weight.shape[4]

    OUT_D  = IN_D - KD + 1
    OUT_H  = IN_H - KH + 1
    OUT_W  = IN_W - KW + 1

    M_total = BATCH * OUT_D * OUT_H * OUT_W

    out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Flatten bias for indexing
    bias_flat = p_bias.view(OUT_C)

    # Strides for x: [BATCH, IN_C, IN_D, IN_H, IN_W]
    stride_xn = x.stride(0)
    stride_xc = x.stride(1)
    stride_xd = x.stride(2)
    stride_xh = x.stride(3)
    stride_xw = x.stride(4)

    # Strides for w: [OUT_C, IN_C, KD, KH, KW]
    stride_wco = p_conv_weight.stride(0)
    stride_wci = p_conv_weight.stride(1)
    stride_wkd = p_conv_weight.stride(2)
    stride_wkh = p_conv_weight.stride(3)
    stride_wkw = p_conv_weight.stride(4)

    BLOCK_M = 256
    BLOCK_N = 32

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    fused_conv3d_activations_kernel[grid](
        x, p_conv_weight, p_conv_bias, bias_flat, out,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wco, stride_wci, stride_wkd, stride_wkh, stride_wkw,
    )

    return out


def get_inputs():
    p_conv_weight = torch.randn(32, 8, 3, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias   = torch.randn(32, device='cuda', dtype=torch.float32)
    p_bias        = torch.randn(32, 1, 1, 1, device='cuda', dtype=torch.float32)
    x             = torch.rand(64, 8, 32, 64, 64, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_conv_weight, p_conv_bias, p_bias, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn.functional as F
        ref_conv = F.conv3d(x, p_conv_weight, p_conv_bias)
        ref = torch.relu(ref_conv)
        ref = F.leaky_relu(ref, negative_slope=0.01)
        ref = F.gelu(ref)
        ref = torch.sigmoid(ref)
        ref = ref + p_bias

        out = kernel_function(p_conv_weight, p_conv_bias, p_bias, x)
        torch.cuda.synchronize()

        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")
    else:
        out = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")