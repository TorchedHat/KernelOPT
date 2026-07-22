import torch
import triton
import triton.language as tl
import math

# Max pool 3x3 stride=1 pad=1 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def maxpool3x3_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(2)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(0)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = c_offs < C
    hw_mask = hw_offs < HW

    h_offs = hw_offs // W
    w_offs = hw_offs % W

    max_val = tl.full((BLOCK_C, BLOCK_HW), float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = h_offs[None, :] + kh - 1
            iw = w_offs[None, :] + kw - 1
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            valid = valid & c_mask[:, None] & hw_mask[None, :]
            ptrs = x_ptr + pid_n * C * H * W + c_offs[:, None] * H * W + ih * W + iw
            val = tl.load(ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    out_ptrs = out_ptr + pid_n * C * H * W + c_offs[:, None] * H * W + h_offs[None, :] * W + w_offs[None, :]
    tl.store(out_ptrs, max_val, mask=c_mask[:, None] & hw_mask[None, :])


# 1x1 conv kernel: x(N,Ci,H,W) * w(Co,Ci) + b(Co) -> out(N,Co,H,W)
# Treat as GEMM: (N*H*W, Ci) x (Ci, Co) -> (N*H*W, Co)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m2[:, None] < M) & (offs_n2[None, :] < N)

    b = tl.load(b_ptr + offs_n2, mask=offs_n2 < N, other=0.0)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m2[:, None] * stride_om + offs_n2[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# 3x3 conv with padding=1: x(N,Ci,H,W) * w(Co,Ci,3,3) + b(Co) -> out(N,Co,H,W)
# Implicit GEMM: M=N*H*W, N=Co, K=Ci*3*3
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_total', 'Co', 'Ci', 'KH', 'KW'],
)
@triton.jit
def conv_nxn_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N_batch, Ci, H, W, Co, KH, KW,
    pad_h, pad_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, oh, ow)
    ow = offs_m % W
    oh = (offs_m // W) % H
    batch = offs_m // (H * W)

    TOTAL_K = Ci * KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(TOTAL_K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K -> (ci, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        ci = offs_k // (KH * KW)

        ih = oh[:, None] + kh[None, :] - pad_h
        iw = ow[:, None] + kw[None, :] - pad_w

        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        valid = valid & (offs_m[:, None] < M_total) & (offs_k[None, :] < TOTAL_K)

        x_ptrs = x_ptr + batch[:, None] * Ci * H * W + ci[None, :] * H * W + ih * W + iw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # w shape: (Co, Ci, KH, KW) -> index: co * Ci*KH*KW + ci * KH*KW + kh * KW + kw
        w_ptrs = w_ptr + offs_n[None, :] * (Ci * KH * KW) + ci[:, None] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        w_mask = (offs_n[None, :] < Co) & (offs_k[:, None] < TOTAL_K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m2[:, None] < M_total) & (offs_n2[None, :] < Co)

    b = tl.load(b_ptr + offs_n2, mask=offs_n2 < Co, other=0.0)
    acc = acc + b[None, :]

    # Output: (N_batch, Co, H, W) -> index: batch * Co * H * W + co * H * W + oh * W + ow
    ow2 = offs_m2 % W
    oh2 = (offs_m2 // W) % H
    batch2 = offs_m2 // (H * W)
    out_ptrs = out_ptr + batch2[:, None] * Co * H * W + offs_n2[None, :] * H * W + oh2[:, None] * W + ow2[:, None]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# Cat kernel: concatenate 4 tensors along channel dim
# Each tensor: (N, Ci, H, W), output: (N, C_total, H, W)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def cat4_kernel(
    t0_ptr, t1_ptr, t2_ptr, t3_ptr,
    out_ptr,
    N, C0, C1, C2, C3, HW,
    C_total,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Decode: offs = n * C_total * HW + c * HW + hw
    hw = offs % HW
    c = (offs // HW) % C_total
    n = offs // (C_total * HW)

    # Determine which tensor and local channel
    in_c0 = c
    in_c1 = c - C0
    in_c2 = c - C0 - C1
    in_c3 = c - C0 - C1 - C2

    is_t0 = c < C0
    is_t1 = (c >= C0) & (c < C0 + C1)
    is_t2 = (c >= C0 + C1) & (c < C0 + C1 + C2)
    is_t3 = c >= C0 + C1 + C2

    val0 = tl.load(t0_ptr + n * C0 * HW + in_c0 * HW + hw, mask=mask & is_t0, other=0.0)
    val1 = tl.load(t1_ptr + n * C1 * HW + in_c1 * HW + hw, mask=mask & is_t1, other=0.0)
    val2 = tl.load(t2_ptr + n * C2 * HW + in_c2 * HW + hw, mask=mask & is_t2, other=0.0)
    val3 = tl.load(t3_ptr + n * C3 * HW + in_c3 * HW + hw, mask=mask & is_t3, other=0.0)

    val = tl.where(is_t0, val0, tl.where(is_t1, val1, tl.where(is_t2, val2, val3)))
    tl.store(out_ptr + offs, val, mask=mask)


def kernel_function(
    p_branch1x1_weight, p_branch1x1_bias,
    p_branch3x3_0_weight, p_branch3x3_0_bias,
    p_branch3x3_1_weight, p_branch3x3_1_bias,
    p_branch5x5_0_weight, p_branch5x5_0_bias,
    p_branch5x5_1_weight, p_branch5x5_1_bias,
    p_branch_pool_1_weight, p_branch_pool_1_bias,
):
    N, Ci, H, W = x.shape
    HW = H * W
    M = N * H * W

    # branch1x1: Conv2d(480->192, 1x1)
    Co1 = p_branch1x1_weight.shape[0]
    branch1x1_out = torch.empty(N, Co1, H, W, device=x.device, dtype=x.dtype)
    # x as (M, Ci), w as (Co, Ci) -> treat w transposed: (Ci, Co)
    x_flat = x.reshape(M, Ci)
    w1 = p_branch1x1_weight.reshape(Co1, Ci)
    grid1 = (triton.cdiv(M, 128), triton.cdiv(Co1, 64))
    conv1x1_kernel[grid1](
        x_flat, w1, p_branch1x1_bias, branch1x1_out.reshape(M, Co1),
        M, Co1, Ci,
        Ci, 1,  # stride_wn, stride_wk (w is Co x Ci, so stride_wn=Ci, stride_wk=1)
    )

    # branch3x3_0: Conv2d(480->96, 1x1)
    Co_r3 = p_branch3x3_0_weight.shape[0]
    branch3x3_0_out = torch.empty(N, Co_r3, H, W, device=x.device, dtype=x.dtype)
    w_r3 = p_branch3x3_0_weight.reshape(Co_r3, Ci)
    grid_r3 = (triton.cdiv(M, 128), triton.cdiv(Co_r3, 64))
    conv1x1_kernel[grid_r3](
        x_flat, w_r3, p_branch3x3_0_bias, branch3x3_0_out.reshape(M, Co_r3),
        M, Co_r3, Ci,
        Ci, 1,
        Ci, 1,
        Co_r3, 1,
    )

    # branch3x3_1: Conv2d(96->208, 3x3, pad=1)
    Co3 = p_branch3x3_1_weight.shape[0]
    branch3x3_out = torch.empty(N, Co3, H, W, device=x.device, dtype=x.dtype)
    grid3 = (triton.cdiv(M, 64), triton.cdiv(Co3, 64))
    conv_nxn_kernel[grid3](
        branch3x3_0_out, p_branch3x3_1_weight, p_branch3x3_1_bias, branch3x3_out,
        N, Co_r3, H, W, Co3, 3, 3,
        1, 1,
        M,
    )

    # branch5x5_0: Conv2d(480->16, 1x1)
    Co_r5 = p_branch5x5_0_weight.shape[0]
    branch5x5_0_out = torch.empty(N, Co_r5, H, W, device=x.device, dtype=x.dtype)
    w_r5 = p_branch5x5_0_weight.reshape(Co_r5, Ci)
    grid_r5 = (triton.cdiv(M, 128), triton.cdiv(Co_r5, 64))
    conv1x1_kernel[grid_r5](
        x_flat, w_r5, p_branch5x5_0_bias, branch5x5_0_out.reshape(M, Co_r5),
        M, Co_r5, Ci,
        Ci, 1,
        Ci, 1,
        Co_r5, 1,
    )

    # branch5x5_1: Conv2d(16->48, 5x5, pad=2)
    Co5 = p_branch5x5_1_weight.shape[0]
    branch5x5_out = torch.empty(N, Co5, H, W, device=x.device, dtype=x.dtype)
    grid5 = (triton.cdiv(M, 64), triton.cdiv(Co5, 64))
    conv_nxn_kernel[grid5](
        branch5x5_0_out, p_branch5x5_1_weight, p_branch5x5_1_bias, branch5x5_out,
        N, Co_r5, H, W, Co5, 5, 5,
        2, 2,
        M,
    )

    # max_pool: MaxPool2d(3x3, stride=1, pad=1)
    pool_out = torch.empty(N, Ci, H, W, device=x.device, dtype=x.dtype)
    BLOCK_C = 32
    BLOCK_HW = 128
    grid_pool = (triton.cdiv(HW, BLOCK_HW), triton.cdiv(Ci, BLOCK_C), N)
    maxpool3x3_kernel[grid_pool](
        x, pool_out,
        N, Ci, H, W,
        HW,
    )

    # branch_pool_1: Conv2d(480->64, 1x1) on pool_out
    Co_p = p_branch_pool_1_weight.shape[0]
    branch_pool_out = torch.empty(N, Co_p, H, W, device=x.device, dtype=x.dtype)
    pool_flat = pool_out.reshape(M, Ci)
    w_p = p_branch_pool_1_weight.reshape(Co_p, Ci)
    grid_p = (triton.cdiv(M, 128), triton.cdiv(Co_p, 64))
    conv1x1_kernel[grid_p](
        pool_flat, w_p, p_branch_pool_1_bias, branch_pool_out.reshape(M, Co_p),
        M, Co_p, Ci,
        Ci, 1,
        Ci, 1,
        Co_p, 1,
    )

    # cat along channel dim: [branch1x1(192), branch3x3(208), branch5x5(48), branch_pool(64)] = 512
    C_total = Co1 + Co3 + Co5 + Co_p
    out = torch.empty(N, C_total, H, W, device=x.device, dtype=x.dtype)
    total_elements = N * C_total * HW
    grid_cat = (triton.cdiv(total_elements, 1024),)
    cat4_kernel[grid_cat](
        branch1x1_out, branch3x3_out, branch5x5_out, branch_pool_out,
        out,
        N, Co1, Co3, Co5, Co_p, HW,
        C_total,
        total_elements,
    )

    return out


def get_inputs():
    p_branch1x1_weight = torch.randn(192, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch1x1_bias = torch.randn(192, dtype=torch.float32, device='cuda')
    p_branch3x3_0_weight = torch.randn(96, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch3x3_0_bias = torch.randn(96, dtype=torch.float32, device='cuda')
    p_branch3x3_1_weight = torch.randn(208, 96, 3, 3, dtype=torch.float32, device='cuda')
    p_branch3x3_1_bias = torch.randn(208, dtype=torch.float32, device='cuda')
    p_branch5x5_0_weight = torch.randn(16, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch5x5_0_bias = torch.randn(16, dtype=torch.float32, device='cuda')
    p_branch5x5_1_weight = torch.randn(48, 16, 5, 5, dtype=torch.float32, device='cuda')
    p_branch5x5_1_bias = torch.randn(48, dtype=torch.float32, device='cuda')
    p_branch_pool_1_weight = torch.randn(64, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch_pool_1_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    x = torch.randn(10, 480, 14, 14, dtype=torch.float32, device='cuda')  # smaller for testing
    return [
        p_branch1x1_weight, p_branch1x1_bias,
        p_branch3x3_0_weight, p_branch3x3_0_bias,
        p_branch3x3_1_weight, p_branch3x3_1_bias,
        p_branch5x5_0_weight, p_branch5x5_0_bias,
        p_branch5x5_1_weight, p_branch5x5_1_bias,
        p_branch_pool_1_weight, p_branch_pool_1_bias,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        (p_branch1x1_weight, p_branch1x1_bias,
         p_branch3x3_0_weight, p_branch3x3_0_bias,
         p_branch3x3_1_weight, p_branch3x3_1_bias,
         p_branch5x5_0_weight, p_branch5x5_0_bias,
         p_branch5x5_1_weight, p_branch5x5_1_bias,
         p_branch_pool_1_weight, p_branch_pool_1_bias,
         x) = inputs

        import torch.nn.functional as F
        branch1x1_ref = F.conv2d(x, p_branch1x1_weight, p_branch1x1_bias)
        branch3x3_ref = F.conv2d(
            F.conv2d(x, p_branch3x3_0_weight, p_branch3x3_0_bias),
            p_branch3x3_1_weight, p_branch3x3_1_bias, padding=1
        )
        branch5x5_ref = F.conv2d(
            F.conv2d(x, p_branch5x5_0_weight, p_branch5x5_0_bias),
            p_branch5x5_1_weight, p_branch5x5_1_bias, padding=2
        )
        pool_ref = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_ref = F.conv2d(pool_ref, p_branch_pool_1_weight, p_branch_pool_1_bias)
        ref = torch.cat([branch1x1_ref, branch3x3_ref, branch5x5_ref, branch_pool_ref], dim=1)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")