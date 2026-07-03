import torch
import triton
import triton.language as tl

# Depthwise convolution kernel
# Input in NHWC layout for coalesced access
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'GROUPS'],
)
@triton.jit
def depthwise_conv_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    GROUPS: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # NHWC strides for X
    stride_xn: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr, stride_xc: tl.constexpr,
    # Weight strides (out_c, in_c_per_group, kh, kw) but for depthwise: (out_c, 1, kh, kw)
    stride_woc: tl.constexpr, stride_wic: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    # Output NHWC strides
    stride_on: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr, stride_oc: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0) -> NHW tile, program_id(1) -> OC tile, program_id(2) -> group
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    group = tl.program_id(2)

    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS

    nhw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_h = nh % OUT_H
    idx_n = nh // OUT_H

    idx_oc = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K

    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PAD_H + idx_h * STRIDE_H
        idx_x_w = j - PAD_W + idx_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k + group * GROUP_IN_C

        x_ptrs = (X_ptr +
                  idx_n[:, None] * stride_xn +
                  idx_x_h[:, None] * stride_xh +
                  idx_x_w[:, None] * stride_xw +
                  idx_x_c[None, :] * stride_xc)

        mask_x = (
            (idx_n < BATCH)[:, None] &
            (idx_x_h >= 0)[:, None] &
            (idx_x_h < IN_H)[:, None] &
            (idx_x_w >= 0)[:, None] &
            (idx_x_w < IN_W)[:, None] &
            ((tl.arange(0, BLOCK_K) + k) < GROUP_IN_C)[None, :]
        )

        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: (out_c, in_c_per_group, kh, kw)
        # For depthwise: in_c_per_group=1, so weight indexed by (group*GROUP_OUT_C + oc, 0, i, j)
        local_k = tl.arange(0, BLOCK_K) + k
        w_ptrs = (W_ptr +
                  (group * GROUP_OUT_C + idx_oc)[None, :] * stride_woc +
                  local_k[:, None] * stride_wic +

        mask_w = (
            (local_k < GROUP_IN_C)[:, None] &
            (idx_oc < GROUP_OUT_C)[None, :]
        )

        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc = tl.dot(matrix_x, matrix_w, acc, allow_tf32=False)

    # Store output in NHWC
    out_ptrs = (Out_ptr +
                idx_n[:, None] * stride_on +
                idx_h[:, None] * stride_oh +
                idx_w[:, None] * stride_ow +
                (group * GROUP_OUT_C + idx_oc)[None, :] * stride_oc)

    mask_out = (
        (idx_n < BATCH)[:, None] &
        (idx_h < OUT_H)[:, None] &
        (idx_w < OUT_W)[:, None] &
        (idx_oc < GROUP_OUT_C)[None, :]
    )

    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# Pointwise conv as GEMM: (M, K) x (K, N) where M=N*H*W, K=in_c, N=out_c
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def pointwise_conv_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_a = (rm[:, None] < M) & (rk[None, :] < k_remaining)
        mask_b = (rk[:, None] < k_remaining) & (rn[None, :] < N)
        a = tl.load(A, mask=mask_a, other=0.0)
        b = tl.load(B, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (rm2[:, None] < M) & (rn2[None, :] < N)
    C_ptrs = C_ptr + rm2[:, None] * stride_cm + rn2[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask_c)


def kernel_function(depthwise_weight, pointwise_weight, x):
    # x: [16, 64, 512, 512] NCHW
    # depthwise_weight: [64, 1, 3, 3]
    # pointwise_weight: [128, 64, 1, 1]

    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C_DW = depthwise_weight.shape[0]  # 64
    GROUPS = IN_C  # depthwise: groups = in_channels = 64
    KERNEL_H, KERNEL_W = depthwise_weight.shape[2], depthwise_weight.shape[3]
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 1, 1

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [16, 512, 512, 64]

    # Output in NHWC
    dw_out_nhwc = torch.empty((BATCH, OUT_H, OUT_W, OUT_C_DW), device=x.device, dtype=x.dtype)

    M_total = BATCH * OUT_H * OUT_W
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 16

    grid = (
        triton.cdiv(M_total, BLOCK_M),
        triton.cdiv(OUT_C_DW // GROUPS, BLOCK_N),
        GROUPS,
    )

    depthwise_conv_kernel[grid](
        x_nhwc, depthwise_weight, dw_out_nhwc,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C_DW, OUT_H, OUT_W,
        GROUPS,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        # NHWC strides for x
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # Weight strides
        depthwise_weight.stride(0), depthwise_weight.stride(1), depthwise_weight.stride(2), depthwise_weight.stride(3),
        # Output NHWC strides
        dw_out_nhwc.stride(0), dw_out_nhwc.stride(1), dw_out_nhwc.stride(2), dw_out_nhwc.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # dw_out_nhwc: [16, 512, 512, 64] NHWC
    # For pointwise conv: treat as (M, K) = (16*512*512, 64) matrix
    # pointwise_weight: [128, 64, 1, 1] -> reshape to [128, 64] then transpose to [64, 128]
    M_pw = BATCH * OUT_H * OUT_W  # 4194304
    K_pw = OUT_C_DW  # 64
    N_pw = pointwise_weight.shape[0]  # 128

    # dw_out_nhwc reshaped to (M_pw, K_pw)
    A = dw_out_nhwc.reshape(M_pw, K_pw)  # (4194304, 64), stride (64, 1)

    # pointwise_weight: [128, 64, 1, 1] -> [128, 64] -> transpose to [64, 128]
    B = pointwise_weight.reshape(N_pw, K_pw).t().contiguous()  # (64, 128)

    # Output: (M_pw, N_pw) in NHWC order, then reshape to NCHW
    pw_out_flat = torch.empty((M_pw, N_pw), device=x.device, dtype=x.dtype)

    grid_pw = lambda meta: (triton.cdiv(M_pw, meta['BLOCK_M']) * triton.cdiv(N_pw, meta['BLOCK_N']),)

    pointwise_conv_gemm_kernel[grid_pw](
        A, B, pw_out_flat,
        M_pw, N_pw, K_pw,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        pw_out_flat.stride(0), pw_out_flat.stride(1),
    )

    # pw_out_flat: (M_pw, N_pw) = (16*512*512, 128)
    # Reshape to NCHW: (16, 512, 512, 128) -> (16, 128, 512, 512)
    out = pw_out_flat.reshape(BATCH, OUT_H, OUT_W, N_pw).permute(0, 3, 1, 2).contiguous()

    return out


def get_inputs():
    depthwise_weight = torch.randn(64, 1, 3, 3, device='cuda', dtype=torch.float32)
    pointwise_weight = torch.randn(128, 64, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.randn(16, 64, 512, 512, device='cuda', dtype=torch.float32)
    return [depthwise_weight, pointwise_weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()

    if not args.profile_only:
        # Verify correctness
        import torch.nn as nn
        dw_weight, pw_weight, x = inputs

        # Reference implementation
        depthwise = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64, bias=False).cuda()
        pointwise = nn.Conv2d(64, 128, 1, bias=False).cuda()
        depthwise.weight.data.copy_(dw_weight)
        pointwise.weight.data.copy_(pw_weight)

        with torch.no_grad():
            ref_out = pointwise(depthwise(x))

        our_out = kernel_function(*inputs)

        print(f"Max diff: {(our_out - ref_out).abs().max().item()}")
        print(f"Mean diff: {(our_out - ref_out).abs().mean().item()}")
        assert torch.allclose(our_out, ref_out, atol=1e-3), "Results don't match!"
        print("Correctness check PASSED!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")