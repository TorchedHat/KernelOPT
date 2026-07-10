import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

# ConvTranspose3d kernel using implicit GEMM approach
# For stride=1, pad=1, kernel=3: output_size = input_size
# ConvTranspose3d(x, w) = Conv3d(x, flip(w)) with appropriate padding

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # X shape: (batch, in_c, in_d, in_h, in_w) - NCDHW
    # W shape: (in_c, out_c, kd, kh, kw) - for ConvTranspose3d
    # Y shape: (batch, out_c, out_d, out_h, out_w)
    batch, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    # X strides
    sx_b, sx_c, sx_d, sx_h, sx_w,
    # W strides: (in_c, out_c, kd, kh, kw)
    sw_ic, sw_oc, sw_kd, sw_kh, sw_kw,
    # Y strides
    sy_b, sy_c, sy_d, sy_h, sy_w,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = batch * OUT_D * OUT_H * OUT_W
    # N = OUT_C
    # K = IN_C * KD * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (b, od, oh, ow)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    od = tmp2 % OUT_D
    b = tmp2 // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (ic, kd, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        tmp_k2 = tmp_k // KH
        kd_idx = tmp_k2 % KD
        ic_idx = tmp_k2 // KD

        # For ConvTranspose3d: input position that contributes to output (od, oh, ow)
        # od = id * stride_d - pad_d + kd => id = (od + pad_d - kd) / stride_d
        # Only valid if (od + pad_d - kd) % stride_d == 0
        id_num_d = od[:, None] + pad_d - kd_idx[None, :]
        id_num_h = oh[:, None] + pad_h - kh_idx[None, :]
        id_num_w = ow[:, None] + pad_w - kw_idx[None, :]

        valid_d = (id_num_d % stride_d) == 0
        valid_h = (id_num_h % stride_h) == 0
        valid_w = (id_num_w % stride_w) == 0

        id_idx = id_num_d // stride_d
        ih_idx = id_num_h // stride_h
        iw_idx = id_num_w // stride_w

        in_bounds = (
            (id_idx >= 0) & (id_idx < IN_D) &
            (ih_idx >= 0) & (ih_idx < IN_H) &
            (iw_idx >= 0) & (iw_idx < IN_W) &
            (ic_idx[None, :] < IN_C) &
            (offs_k[None, :] < K) &
            (offs_m[:, None] < M)
        )

        x_ptrs = (X_ptr +
                  b[:, None] * sx_b +
                  ic_idx[None, :] * sx_c +
        x_tile = tl.load(x_ptrs, mask=in_bounds, other=0.0)

        # Weight: w[ic, oc, kd, kh, kw]
        # For ConvTranspose3d, weight is indexed as w[ic, oc, kd, kh, kw]
        w_ptrs = (W_ptr +
                  ic_idx[:, None] * sw_ic +
                  offs_n[None, :] * sw_oc +
                  kd_idx[:, None] * sw_kd +
                  kh_idx[:, None] * sw_kh +
                  kw_idx[:, None] * sw_kw)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = (Y_ptr +
              b[:, None] * sy_b +
              offs_n[None, :] * sy_c +
              od[:, None] * sy_d +
              oh[:, None] * sy_h +
              ow[:, None] * sy_w)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 1}, num_warps=8, num_stages=2),
    ],
    key=['C', 'D', 'HW'],
)
@triton.jit
def fused_mean_bias_softmax_tanh_scale_kernel(
    B, C, D, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    D_VAL: tl.constexpr,
):
    # Each program handles one (batch, hw_pos) pair
    # Reduces over C for softmax, reduces over D for mean
    pid = tl.program_id(0)
    total = B * HW
    if pid >= total:
        return

    b_idx = pid // HW
    hw_idx = pid % HW

    # For each channel, compute mean over D, add bias
    # Then softmax over channels, tanh, scale

    offs_c = tl.arange(0, BLOCK_C)

    # Step 1: Compute mean over D for each channel
    # mean[b, c, hw] = sum_d(Y[b, c, d, hw]) / D
    mean_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for d in range(D_VAL):
        y_ptrs = (Y_ptr +
        mask_c = offs_c < C
        y_vals = tl.load(y_ptrs, mask=mask_c, other=0.0)
        mean_vals = mean_vals + y_vals

    mean_vals = mean_vals / D_VAL

    # Step 2: Add bias
    bias_vals = tl.load(Bias_ptr + offs_c, mask=offs_c < C, other=0.0)
    mean_vals = mean_vals + bias_vals

    # Step 3: Softmax over channels
    # Find max for numerical stability
    c_max = tl.max(mean_vals, axis=0)
    exp_vals = tl.exp(mean_vals - c_max)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / exp_sum

    # Step 4: Tanh
    tanh_vals = libdevice.tanh(softmax_vals)

    # Step 5: Scale
    out_vals = tanh_vals * scaling_factor

    # Store output: (B, C, 1, H, W) -> treating as (B, C, HW)
    out_ptrs = Out_ptr + b_idx * so_b + offs_c * so_c + hw_idx * so_hw
    tl.store(out_ptrs, out_vals, mask=offs_c < C)


def kernel_function(weight, bias_conv, bias_param, x):
    # weight: (in_c=16, out_c=64, kd=3, kh=3, kw=3) - ConvTranspose3d weight
    # bias_conv: (64,)
    # bias_param: (1, 64, 1, 1, 1)
    # x: (16, 16, 32, 128, 128)

    B, IN_C, IN_D, IN_H, IN_W = x.shape
    # ConvTranspose3d weight shape: (in_channels, out_channels, kD, kH, kW)
    _, OUT_C, KD, KH, KW = weight.shape
    stride_d = stride_h = stride_w = 1
    pad_d = pad_h = pad_w = 1

    OUT_D = (IN_D - 1) * stride_d - 2 * pad_d + KD  # = 32
    OUT_H = (IN_H - 1) * stride_h - 2 * pad_h + KH  # = 128
    OUT_W = (IN_W - 1) * stride_w - 2 * pad_w + KW  # = 128

    # Allocate output for conv_transpose3d
    y_conv = torch.empty(B, OUT_C, OUT_D, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    M = B * OUT_D * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KD * KH * KW

    # X strides: (B, IN_C, IN_D, IN_H, IN_W)
    sx_b, sx_c, sx_d, sx_h, sx_w = x.stride()
    # W strides: (IN_C, OUT_C, KD, KH, KW)
    sw_ic, sw_oc, sw_kd, sw_kh, sw_kw = weight.stride()
    # Y strides: (B, OUT_C, OUT_D, OUT_H, OUT_W)
    sy_b, sy_c, sy_d, sy_h, sy_w = y_conv.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid](
        x, weight, bias_conv, y_conv,
        B, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        sx_b, sx_c, sx_d, sx_h, sx_w,
        sw_ic, sw_oc, sw_kd, sw_kh, sw_kw,
        sy_b, sy_c, sy_d, sy_h, sy_w,
        M, N, K,
    )

    # Now fuse: mean(dim=2) + add(bias) + softmax(dim=1) + tanh + scale
    # y_conv: (B, OUT_C, OUT_D, OUT_H, OUT_W)
    # Output: (B, OUT_C, 1, OUT_H, OUT_W)
    HW = OUT_H * OUT_W
    out = torch.empty(B, OUT_C, 1, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    # bias_param: (1, 64, 1, 1, 1) -> reshape to (64,)
    bias_flat = bias_param.reshape(OUT_C)

    # y_conv strides for the fused kernel
    # We treat H*W as a single dimension
    # y_conv: (B, C, D, H, W) with strides (sy_b, sy_c, sy_d, sy_h, sy_w)
    # hw_idx = h * OUT_W + w, so stride for hw is sy_w=1, and h contributes sy_h=OUT_W
    # We pass sy_h as the stride for hw_idx (since hw_idx = h*OUT_W + w)

    # out strides: (B, C, 1, H, W)
    so_b, so_c, so_1, so_h, so_w = out.stride()
    # so_hw = so_w = 1 (since W is contiguous)

    total_programs = B * HW
    grid2 = (total_programs,)

    fused_mean_bias_softmax_tanh_scale_kernel[grid2](
        y_conv, bias_flat, out,
        B, OUT_C, OUT_D, HW,
        sy_b, sy_c, sy_d, sy_h,
        so_b, so_c, so_w,
        2.0,  # scaling_factor
        D_VAL=OUT_D,
    )

    return out


def get_inputs():
    torch.manual_seed(42)
    weight = torch.randn(16, 64, 3, 3, 3, dtype=torch.float32, device='cuda')
    bias_conv = torch.randn(64, dtype=torch.float32, device='cuda')
    bias_param = torch.randn(1, 64, 1, 1, 1, dtype=torch.float32, device='cuda')
    x = torch.rand(16, 16, 32, 128, 128, dtype=torch.float32, device='cuda')
    return [weight, bias_conv, bias_param, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, bias_conv, bias_param, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        # Reference
        conv_t = nn.ConvTranspose3d(16, 64, 3, stride=1, padding=1).cuda()
        with torch.no_grad():
            conv_t.weight.copy_(weight)
            conv_t.bias.copy_(bias_conv)
        bias_p = nn.Parameter(bias_param.clone())

        with torch.no_grad():
            ref = conv_t(x)
            ref = ref.mean(dim=2, keepdim=True)
            ref = ref + bias_p
            ref = torch.softmax(ref, dim=1)
            ref = torch.tanh(ref)
            ref = ref * 2.0

        out = kernel_function(weight, bias_conv, bias_param, x)
        print(f"Max diff: {(out - ref).abs().max().item()}")
        print(f"Mean diff: {(out - ref).abs().mean().item()}")
        assert (out - ref).abs().max().item() < 1e-2, "Results don't match!"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")