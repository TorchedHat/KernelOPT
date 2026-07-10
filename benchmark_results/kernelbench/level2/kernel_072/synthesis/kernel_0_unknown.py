import torch
import triton
import triton.language as tl
import math

# ConvTranspose3d kernel: for each output position, scatter from input
# We implement it as: for each output element, gather from input
# Output[n, co, od, oh, ow] = sum over (ci, kd, kh, kw) of:
#   input[n, ci, id, ih, iw] * weight[ci, co, kd, kh, kw]
# where id = (od + padding - kd) / stride (only if divisible)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 8}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, CI, CO,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    # strides for X: (N, CI, ID, IH, IW)
    sx_n, sx_ci, sx_d, sx_h, sx_w,
    # strides for W: (CI, CO, KD, KH, KW)
    sw_ci, sw_co, sw_kd, sw_kh, sw_kw,
    # strides for Y: (N, CO, OD, OH, OW)
    sy_n, sy_co, sy_d, sy_h, sy_w,
    M, K,  # M = N*OD*OH*OW, K = CI*KD*KH*KW
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: n, od, oh, ow
    OHW = OH * OW
    ODHW = OD * OH * OW
    n_idx = offs_m // ODHW
    rem = offs_m % ODHW
    od_idx = rem // OHW
    rem2 = rem % OHW
    oh_idx = rem2 // OW
    ow_idx = rem2 % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K index: ci, kd, kh, kw
        KHKW = KH * KW
        KDKHKW = KD * KH * KW
        ci_idx = offs_k // KDKHKW
        rem_k = offs_k % KDKHKW
        kd_idx = rem_k // KHKW
        rem_k2 = rem_k % KHKW
        kh_idx = rem_k2 // KW
        kw_idx = rem_k2 % KW

        # For ConvTranspose: output[od] comes from input[id] where
        # od = id * stride - pad + k  =>  id = (od + pad - k) / stride
        # Only valid if (od + pad - k) % stride == 0 and 0 <= id < ID
        od_plus_pad_d = od_idx[:, None] + PAD_D - kd_idx[None, :]
        od_plus_pad_h = oh_idx[:, None] + PAD_H - kh_idx[None, :]
        od_plus_pad_w = ow_idx[:, None] + PAD_W - kw_idx[None, :]

        valid_d = (od_plus_pad_d % STRIDE_D == 0)
        valid_h = (od_plus_pad_h % STRIDE_H == 0)
        valid_w = (od_plus_pad_w % STRIDE_W == 0)

        id_idx = od_plus_pad_d // STRIDE_D
        ih_idx = od_plus_pad_h // STRIDE_H
        iw_idx = od_plus_pad_w // STRIDE_W

        in_bounds_d = (id_idx >= 0) & (id_idx < ID)
        in_bounds_h = (ih_idx >= 0) & (ih_idx < IH)
        in_bounds_w = (iw_idx >= 0) & (iw_idx < IW)
        in_bounds_n = (n_idx[:, None] < N)
        in_bounds_k = (offs_k[None, :] < K)
        in_bounds_ci = (ci_idx[None, :] < CI)

        valid_mask = (valid_d & valid_h & valid_w &

        x_ptrs = (X_ptr +
                  n_idx[:, None] * sx_n +
                  ci_idx[None, :] * sx_ci +
        x_tile = tl.load(x_ptrs, mask=valid_mask, other=0.0)

        # Weight: W[ci, co, kd, kh, kw]
        w_ptrs = (W_ptr +
                  ci_idx[:, None] * sw_ci +
                  offs_n[None, :] * sw_co +
                  kd_idx[:, None] * sw_kd +
                  kh_idx[:, None] * sw_kh +
                  kw_idx[:, None] * sw_kw)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < CO)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < CO, other=0.0)
    acc = acc + bias[None, :]

    # Store
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < CO
    mask = m_mask & n_mask

    y_ptrs = (Y_ptr +
              n_idx[:, None] * sy_n +
              offs_n[None, :] * sy_co +
              od_idx[:, None] * sy_d +
              oh_idx[:, None] * sy_h +
              ow_idx[:, None] * sy_w)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask)


# Fused BatchNorm + AvgPool3d(2) + AvgPool3d(2) kernel
# Input: (N, C, D, H, W) = (64, 16, 63, 63, 63)
# After BN: same shape
# After pool1 (k=2, stride=2): (64, 16, 31, 31, 31)
# After pool2 (k=2, stride=2): (64, 16, 15, 15, 15)
# Each output element at (n, c, od, oh, ow) in final output corresponds to
# a 4x4x4 region in the BN output (2x2x2 for pool1, then 2x2x2 for pool2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
    ],
    key=['total_elements'],
)
@triton.jit
def bn_avgpool_kernel(
    X_ptr, Y_ptr,
    BN_weight_ptr, BN_bias_ptr, BN_mean_ptr, BN_var_ptr,
    N, C,
    eps,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Decode output index: (n, c, od, oh, ow)
    OHW = OH * OW
    ODHW = OD * OH * OW
    CODHW = C * ODHW

    n_idx = offs // CODHW
    rem = offs % CODHW
    c_idx = rem // ODHW
    rem2 = rem % ODHW
    od_idx = rem2 // OHW
    rem3 = rem2 % OHW
    oh_idx = rem3 // OW
    ow_idx = rem3 % OW

    # Load BN params for channel c
    bn_mean = tl.load(BN_mean_ptr + c_idx, mask=mask, other=0.0)
    bn_var = tl.load(BN_var_ptr + c_idx, mask=mask, other=1.0)
    bn_w = tl.load(BN_weight_ptr + c_idx, mask=mask, other=1.0)
    bn_b = tl.load(BN_bias_ptr + c_idx, mask=mask, other=0.0)
    bn_std = tl.sqrt(bn_var + eps)

    # Each output (od, oh, ow) in final output corresponds to
    # input region [od*4 : od*4+4, oh*4 : oh*4+4, ow*4 : ow*4+4]
    # (two 2x2x2 pools = 4x4x4 region, 64 elements)
    # Average over 64 elements

    # Input strides: (N, C, ID, IH, IW) contiguous
    IHW = IH * IW
    IDHW = ID * IH * IW
    CIDHW = C * IDHW

    acc = tl.zeros_like(bn_mean)  # float32 accumulator

    # Unroll the 4x4x4 = 64 element sum
    # Base input coordinates
    id_base = od_idx * 4
    ih_base = oh_idx * 4
    iw_base = ow_idx * 4

    # We'll loop over the 64 elements
    for dd in range(4):
        for dh in range(4):
            for dw in range(4):
                id_i = id_base + dd
                ih_i = ih_base + dh
                iw_i = iw_base + dw
                in_bounds = (id_i < ID) & (ih_i < IH) & (iw_i < IW) & mask
                x_ptr = (X_ptr +
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)
                # Apply BN
                x_norm = (x_val - bn_mean) / bn_std
                x_bn = x_norm * bn_w + bn_b
                acc = acc + x_bn

    # Divide by 64 (4x4x4)
    result = acc / 64.0

    # Store output
    y_ptr = (Y_ptr +
    tl.store(y_ptr, result, mask=mask)


def kernel_function(
    p_conv_transpose_weight,
    p_conv_transpose_bias,
    p_batch_norm_weight,
    p_batch_norm_bias,
    b_batch_norm_running_mean,
    b_batch_norm_running_var,
    x,
):
    # x: (64, 3, 32, 32, 32)
    # weight: (CI=3, CO=16, KD=3, KH=3, KW=3)
    N, CI, ID, IH, IW = x.shape
    CI_w, CO, KD, KH, KW = p_conv_transpose_weight.shape

    STRIDE_D, STRIDE_H, STRIDE_W = 2, 2, 2
    PAD_D, PAD_H, PAD_W = 1, 1, 1

    OD = (ID - 1) * STRIDE_D - 2 * PAD_D + KD  # = 63
    OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH  # = 63
    OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW  # = 63

    # Allocate conv output
    conv_out = torch.empty((N, CO, OD, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OD * OH * OW
    K_total = CI * KD * KH * KW

    # X strides
    sx_n, sx_ci, sx_d, sx_h, sx_w = (
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4)
    )
    # W strides: (CI, CO, KD, KH, KW)
    w = p_conv_transpose_weight
    sw_ci, sw_co, sw_kd, sw_kh, sw_kw = (
        w.stride(0), w.stride(1), w.stride(2), w.stride(3), w.stride(4)
    )
    # Y strides
    sy_n, sy_co, sy_d, sy_h, sy_w = (
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4)
    )

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(CO, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid](
        x, w, p_conv_transpose_bias, conv_out,
        N, CI, CO,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        sx_n, sx_ci, sx_d, sx_h, sx_w,
        sw_ci, sw_co, sw_kd, sw_kh, sw_kw,
        sy_n, sy_co, sy_d, sy_h, sy_w,
        M, K_total,
    )

    # Fused BN + AvgPool + AvgPool
    # Output after two 2x2x2 avg pools: (N, CO, 15, 15, 15)
    # floor((63-2)/2+1) = 31, floor((31-2)/2+1) = 15
    pool1_D = (OD) // 2  # 31
    pool1_H = (OH) // 2  # 31
    pool1_W = (OW) // 2  # 31
    pool2_D = pool1_D // 2  # 15
    pool2_H = pool1_H // 2  # 15
    pool2_W = pool1_W // 2  # 15

    final_out = torch.empty((N, CO, pool2_D, pool2_H, pool2_W), device=x.device, dtype=x.dtype)

    total_elements = N * CO * pool2_D * pool2_H * pool2_W
    eps = 1e-5

    grid2 = lambda meta: (cdiv(total_elements, meta['BLOCK_SIZE']),)

    bn_avgpool_kernel[grid2](
        conv_out, final_out,
        p_batch_norm_weight, p_batch_norm_bias,
        b_batch_norm_running_mean, b_batch_norm_running_var,
        N, CO,
        OD, OH, OW,
        pool2_D, pool2_H, pool2_W,
        eps,
        total_elements,
    )

    return final_out


def get_inputs():
    batch_size = 64
    in_channels = 3
    out_channels = 16
    depth, height, width = 32, 32, 32
    kernel_size = 3
    stride = 2
    padding = 1

    torch.manual_seed(42)
    p_conv_transpose_weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
    p_conv_transpose_bias = torch.randn(out_channels)
    p_batch_norm_weight = torch.ones(out_channels)
    p_batch_norm_bias = torch.zeros(out_channels)
    b_batch_norm_running_mean = torch.zeros(out_channels)
    b_batch_norm_running_var = torch.ones(out_channels)
    x = torch.rand(batch_size, in_channels, depth, height, width)

    return [
        p_conv_transpose_weight,
        p_conv_transpose_bias,
        p_batch_norm_weight,
        p_batch_norm_bias,
        b_batch_norm_running_mean,
        b_batch_norm_running_var,
        x,
    ]


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
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 2
        padding = 1

        model = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.AvgPool3d(kernel_size=2),
            nn.AvgPool3d(kernel_size=2),
        )

        # Set weights
        with torch.no_grad():
            model[0].weight.copy_(inputs[0])
            model[0].bias.copy_(inputs[1])
            model[1].weight.copy_(inputs[2])
            model[1].bias.copy_(inputs[3])
            model[1].running_mean.copy_(inputs[4])
            model[1].running_var.copy_(inputs[5])
            model[1].eval()

        model.eval()
        x_cpu = inputs[6]
        ref_out = model(x_cpu)

        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref_out.shape}")
        max_diff = (output.cpu() - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-2:
            print("PASSED!")
        else:
            print("FAILED!")