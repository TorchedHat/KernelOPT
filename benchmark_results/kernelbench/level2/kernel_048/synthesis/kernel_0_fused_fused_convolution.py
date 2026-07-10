import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_D', 'IN_H', 'IN_W', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W',
         'KERNEL_D', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv3d_epilogue_kernel(
    X_ptr, W_ptr, bias_ptr, scale_ptr, bias2_ptr, out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_D: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xd: tl.constexpr,
    stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_wco: tl.constexpr, stride_wci: tl.constexpr, stride_wd: tl.constexpr,
    stride_wh: tl.constexpr, stride_ww: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_d, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    out_d_idx = tmp2 % OUT_D
    batch_idx = tmp2 // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_D * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(BLOCK_K_COUNT):
        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kd, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        tmp_k2 = tmp_k // KERNEL_H
        kd_idx = tmp_k2 % KERNEL_D
        ic_idx = tmp_k2 // KERNEL_D

        # Input coordinates
        in_d_coord = out_d_idx[:, None] + kd_idx[None, :]
        in_h_coord = out_h_idx[:, None] + kh_idx[None, :]
        in_w_coord = out_w_idx[:, None] + kw_idx[None, :]

        valid_m = (batch_idx < BATCH) & (out_d_idx < OUT_D) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
        valid_k = offs_k < TOTAL_K
        valid_in = (in_d_coord >= 0) & (in_d_coord < IN_D) & \
                   (in_h_coord >= 0) & (in_h_coord < IN_H) & \
                   (in_w_coord >= 0) & (in_w_coord < IN_W) & \
                   (ic_idx[None, :] < IN_C)

        x_mask = valid_m[:, None] & valid_in & valid_k[None, :]

        x_ptrs = X_ptr + (
            batch_idx[:, None] * stride_xn +
            ic_idx[None, :] * stride_xc +
        )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: shape (OUT_C, IN_C, KD, KH, KW)
        # stride: (stride_wco, stride_wci, stride_wd, stride_wh, stride_ww)
        w_ptrs = W_ptr + (
            offs_n[None, :] * stride_wco +
            ic_idx[:, None] * stride_wci +
            kd_idx[:, None] * stride_wd +
            kh_idx[:, None] * stride_wh +
            kw_idx[:, None] * stride_ww
        )
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Epilogue: add bias, scale, tanh, bias2, sigmoid
    valid_out_m = (batch_idx < BATCH) & (out_d_idx < OUT_D) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
    valid_out_n = offs_n < OUT_C
    out_mask = valid_out_m[:, None] & valid_out_n[None, :]

    # Load bias (per output channel)
    bias_vals = tl.load(bias_ptr + offs_n, mask=valid_out_n, other=0.0)
    acc = acc + bias_vals[None, :]

    # Load scaling_factor (per output channel)
    scale_vals = tl.load(scale_ptr + offs_n, mask=valid_out_n, other=0.0)
    acc = acc * scale_vals[None, :]

    # tanh
    acc = libdevice.tanh(acc)

    # Load bias2 (per output channel)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=valid_out_n, other=0.0)
    acc = acc * bias2_vals[None, :]

    # sigmoid
    acc = tl.sigmoid(acc)

    # Store output
    # Output layout: (BATCH, OUT_C, OUT_D, OUT_H, OUT_W)
    # stride: (OUT_C*OUT_D*OUT_H*OUT_W, OUT_D*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    out_stride_n = OUT_C * OUT_D * OUT_H * OUT_W
    out_stride_c = OUT_D * OUT_H * OUT_W
    out_stride_d = OUT_H * OUT_W
    out_stride_h = OUT_W
    out_stride_w = 1

    out_ptrs = out_ptr + (
        batch_idx[:, None] * out_stride_n +
        offs_n[None, :] * out_stride_c +
        out_d_idx[:, None] * out_stride_d +
        out_h_idx[:, None] * out_stride_h +
        out_w_idx[:, None] * out_stride_w
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_conv_weight, p_conv_bias, p_scaling_factor, p_bias, x):
    # x: (128, 3, 16, 64, 64)
    # p_conv_weight: (16, 3, 3, 3, 3)
    # p_conv_bias: (16,)
    # p_scaling_factor: (16, 1, 1, 1)
    # p_bias: (16, 1, 1, 1)

    BATCH = x.shape[0]
    IN_C = x.shape[1]
    IN_D = x.shape[2]
    IN_H = x.shape[3]
    IN_W = x.shape[4]

    OUT_C = p_conv_weight.shape[0]
    KERNEL_D = p_conv_weight.shape[2]
    KERNEL_H = p_conv_weight.shape[3]
    KERNEL_W = p_conv_weight.shape[4]

    # No padding, stride=1
    OUT_D = IN_D - KERNEL_D + 1
    OUT_H = IN_H - KERNEL_H + 1
    OUT_W = IN_W - KERNEL_W + 1

    out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Flatten scaling_factor and bias to 1D
    scale_1d = p_scaling_factor.reshape(OUT_C)
    bias2_1d = p_bias.reshape(OUT_C)

    M = BATCH * OUT_D * OUT_H * OUT_W
    N = OUT_C

    # Strides for input x
    stride_xn = x.stride(0)
    stride_xc = x.stride(1)
    stride_xd = x.stride(2)
    stride_xh = x.stride(3)
    stride_xw = x.stride(4)

    # Strides for weight
    stride_wco = p_conv_weight.stride(0)
    stride_wci = p_conv_weight.stride(1)
    stride_wd = p_conv_weight.stride(2)
    stride_wh = p_conv_weight.stride(3)
    stride_ww = p_conv_weight.stride(4)

    # We need to pick BLOCK_M and BLOCK_N for the grid
    # Use fixed values matching autotune configs
    BLOCK_N_FIXED = 16

    def grid(meta):
        bm = meta['BLOCK_M']
        bn = meta['BLOCK_N']
        return (triton.cdiv(M, bm), triton.cdiv(N, bn))

    conv3d_epilogue_kernel[grid](
        x, p_conv_weight, p_conv_bias, scale_1d, bias2_1d, out,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KERNEL_D, KERNEL_H, KERNEL_W,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wco, stride_wci, stride_wd, stride_wh, stride_ww,
    )

    return out


def get_inputs():
    p_conv_weight = torch.randn(16, 3, 3, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(16, device='cuda', dtype=torch.float32)
    p_scaling_factor = torch.randn(16, 1, 1, 1, device='cuda', dtype=torch.float32)
    p_bias = torch.randn(16, 1, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 3, 16, 64, 64, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, p_scaling_factor, p_bias, x]


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
        p_conv_weight, p_conv_bias, p_scaling_factor, p_bias, x = inputs

        # Reference computation
        conv_out = F.conv3d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        ref = conv_out * p_scaling_factor
        ref = torch.tanh(ref)
        ref = ref * p_bias
        ref = torch.sigmoid(ref)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")