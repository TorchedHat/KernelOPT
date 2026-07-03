import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, Y_ptr,
    # Input dimensions
    N, CI, ID, IH, IW,
    # Output dimensions
    CO, OD, OH, OW,
    # Kernel dimensions
    KD, KH, KW,
    # Stride, padding
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    # Strides for X (N, CI, ID, IH, IW)
    x_sn, x_sc, x_sd, x_sh, x_sw,
    # Strides for W (CI, CO, KD, KH, KW)
    w_sci, w_sco, w_skd, w_skh, w_skw,
    # Strides for Y (N, CO, OD, OH, OW)
    y_sn, y_sc, y_sd, y_sh, y_sw,
    # Problem sizes for GEMM
    M, N_out, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    
    M = N * OD * OH * OW  (output spatial positions)
    N_out = CO             (output channels)
    K = CI * KD * KH * KW (reduction: input channels * kernel volume)
    
    Y[n, co, od, oh, ow] = sum over (ci, kd, kh, kw) of:
        X[n, ci, id, ih, iw] * W[ci, co, kd, kh, kw]
    where id = (od - kd + PAD_D) / STRIDE_D (only if divisible and in bounds)
          ih = (oh - kh + PAD_H) / STRIDE_H
          iw = (ow - kw + PAD_W) / STRIDE_W
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode output spatial positions from linear M index
    # M index = n * (OD*OH*OW) + od * (OH*OW) + oh * OW + ow
    OHW = OH * OW
    ODHW = OD * OH * OW

    m_n = offs_m // ODHW
    m_rem = offs_m % ODHW
    m_od = m_rem // OHW
    m_rem2 = m_rem % OHW
    m_oh = m_rem2 // OW
    m_ow = m_rem2 % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index: K = CI * KD * KH * KW
        # k = ci * (KD*KH*KW) + kd * (KH*KW) + kh * KW + kw
        KHW = KH * KW
        KDHW = KD * KH * KW

        k_ci = offs_k // KDHW
        k_rem = offs_k % KDHW
        k_kd = k_rem // KHW
        k_rem2 = k_rem % KHW
        k_kh = k_rem2 // KW
        k_kw = k_rem2 % KW

        # For ConvTranspose3d:
        # od = id * STRIDE_D + kd - PAD_D
        # => id = (od - kd + PAD_D) / STRIDE_D (must be integer and in [0, ID))
        # 
        # For each (m, k) pair:
        # od = m_od[m], kd = k_kd[k]
        # id_num = od - kd + PAD_D  (must be divisible by STRIDE_D and in range)

        od_m = m_od[:, None]  # (BLOCK_M, 1)
        oh_m = m_oh[:, None]
        ow_m = m_ow[:, None]
        n_m = m_n[:, None]

        kd_k = k_kd[None, :]  # (1, BLOCK_K)
        kh_k = k_kh[None, :]
        kw_k = k_kw[None, :]
        ci_k = k_ci[None, :]

        # Compute input coordinates
        id_num = od_m - kd_k + PAD_D
        ih_num = oh_m - kh_k + PAD_H
        iw_num = ow_m - kw_k + PAD_W

        # Check divisibility by stride
        id_valid_stride = (id_num % STRIDE_D) == 0
        ih_valid_stride = (ih_num % STRIDE_H) == 0
        iw_valid_stride = (iw_num % STRIDE_W) == 0

        id_idx = id_num // STRIDE_D
        ih_idx = ih_num // STRIDE_H
        iw_idx = iw_num // STRIDE_W

        # Bounds checks
        valid_id = id_valid_stride & (id_idx >= 0) & (id_idx < ID)
        valid_ih = ih_valid_stride & (ih_idx >= 0) & (ih_idx < IH)
        valid_iw = iw_valid_stride & (iw_idx >= 0) & (iw_idx < IW)
        valid_ci = ci_k < CI
        valid_k = offs_k[None, :] < K
        valid_m = offs_m[:, None] < M

        valid_x = valid_m & valid_id & valid_ih & valid_iw & valid_ci & valid_k

        # Load X tile: shape (BLOCK_M, BLOCK_K)
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Load W tile: shape (BLOCK_K, BLOCK_N)
        # W[ci, co, kd, kh, kw] -> for each k=(ci,kd,kh,kw), co in offs_n
        offs_n_w = offs_n[None, :]  # (1, BLOCK_N)
        valid_n = offs_n_w < CO
        valid_w = (offs_k[:, None] < K) & valid_n

        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store output
    valid_out_m = offs_m[:, None] < M
    valid_out_n = offs_n[None, :] < CO

    out_n = m_n[:, None]
    out_od = m_od[:, None]
    out_oh = m_oh[:, None]
    out_ow = m_ow[:, None]
    out_co = offs_n[None, :]

    y_ptrs = (Y_ptr

    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out_m & valid_out_n)


def kernel_function(weight, x):
    """
    
    weight: (CI, CO, KD, KH, KW) = (48, 48, 3, 3, 3)
    x: (N, CI, ID, IH, IW) = (8, 48, 64, 64, 64)
    
    With stride=1, padding=0, kernel=3:
    OD = (ID - 1) * stride - 2*pad + KD = 63 + 3 = 66
    """
    N, CI, ID, IH, IW = x.shape
    CI_w, CO, KD, KH, KW = weight.shape

    assert CI == CI_w, f"Channel mismatch: {CI} vs {CI_w}"

    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0

    OD = (ID - 1) * STRIDE_D - 2 * PAD_D + KD
    OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH
    OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW

    # Allocate output
    y = torch.zeros(N, CO, OD, OH, OW, device=x.device, dtype=x.dtype)

    M = N * OD * OH * OW
    N_out = CO
    K = CI * KD * KH * KW

    # Make tensors contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    # Strides for X (N, CI, ID, IH, IW)
    x_sn, x_sc, x_sd, x_sh, x_sw = x.stride()

    # Strides for W (CI, CO, KD, KH, KW)
    w_sci, w_sco, w_skd, w_skh, w_skw = weight.stride()

    # Strides for Y (N, CO, OD, OH, OW)
    y_sn, y_sc, y_sd, y_sh, y_sw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N_out, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid](
        x, weight, y,
        N, CI, ID, IH, IW,
        CO, OD, OH, OW,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        x_sn, x_sc, x_sd, x_sh, x_sw,
        w_sci, w_sco, w_skd, w_skh, w_skw,
        y_sn, y_sc, y_sd, y_sh, y_sw,
        M, N_out, K,
    )

    return y


def get_inputs():
    weight = torch.randn(48, 48, 3, 3, 3)
    x = torch.randn(8, 48, 64, 64, 64)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_gpu = [t.cuda() for t in inputs]

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        weight_gpu, x_gpu = inputs_gpu

        # Reference: PyTorch ConvTranspose3d
        # Note: nn.ConvTranspose3d weight shape is (in_channels, out_channels, kD, kH, kW)
        # which matches our weight shape (CI, CO, KD, KH, KW)
        ref_conv = nn.ConvTranspose3d(48, 48, 3, stride=1, padding=0, bias=False).cuda()
        with torch.no_grad():
            ref_conv.weight.copy_(weight_gpu)
        ref_out = ref_conv(x_gpu)

        our_out = kernel_function(weight_gpu, x_gpu)

        print(f"Output shape: {our_out.shape}")
        print(f"Reference shape: {ref_out.shape}")
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-3:
            print("PASSED: Results match within tolerance!")
        else:
            print("FAILED: Results do not match!")
            # Debug: check a few values
            print(f"Our output sample: {our_out[0, 0, 0, 0, :5]}")
            print(f"Ref output sample: {ref_out[0, 0, 0, 0, :5]}")
    else:
        output = kernel_function(*inputs_gpu)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
        print(f"Output shape: {output.shape}")