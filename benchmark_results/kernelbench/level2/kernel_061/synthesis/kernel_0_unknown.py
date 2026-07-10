import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 8}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv_transpose3d_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    TOTAL_K,
    M_TOTAL,
    # strides for X: (N, C, D, H, W)
    s_xn, s_xc, s_xd, s_xh, s_xw,
    # strides for W: (IN_C, OUT_C, KD, KH, KW)
    s_wi, s_wo, s_wkd, s_wkh, s_wkw,
    # strides for Y: (N, C, D, H, W)
    s_yn, s_yc, s_yd, s_yh, s_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: n, od, oh, ow
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    od = tmp2 % OUT_D
    n = tmp2 // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K index: ci, kd, kh, kw
        # K = IN_C * KD * KH * KW, ordered as ci * (KD*KH*KW) + kd*(KH*KW) + kh*KW + kw
        kw_k = offs_k % KW
        tmp_k = offs_k // KW
        kh_k = tmp_k % KH
        tmp_k2 = tmp_k // KH
        kd_k = tmp_k2 % KD
        ci_k = tmp_k2 // KD

        # For ConvTranspose3d: input position id = od - kd (no stride, no padding)
        id_ = od[:, None] - kd_k[None, :]
        ih_ = oh[:, None] - kh_k[None, :]
        iw_ = ow[:, None] - kw_k[None, :]

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (id_ >= 0) & (id_ < IN_D) & (ih_ >= 0) & (ih_ < IN_H) & (iw_ >= 0) & (iw_ < IN_W)
        valid_n = n[:, None] < N
        valid_ci = ci_k[None, :] < IN_C
        mask_x = valid_m & valid_k & valid_spatial & valid_n & valid_ci

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (IN_C, OUT_C, KD, KH, KW)
        # offs_n indexes OUT_C
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C) & valid_ci[:, None]
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Store output
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['spatial_size'],
)
@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    N, C, spatial_size,
    num_groups, group_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    # Elements per group = group_size * spatial_size
    elems_per_group = group_size * spatial_size

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, elems_per_group, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        # offs indexes into (c_within_group * spatial_size + spatial_idx)
        c_local = offs // spatial_size
        s_idx = offs % spatial_size
        c_global = g * group_size + c_local
        # spatial_size = D * H * W, decode to d, h, w
        # Actually we can just use flat spatial index
        mask = (offs < elems_per_group) & (c_local < group_size)
        x_ptrs = X_ptr + n * C * spatial_size + c_global * spatial_size + s_idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / elems_per_group

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, elems_per_group, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        c_local = offs // spatial_size
        s_idx = offs % spatial_size
        c_global = g * group_size + c_local
        mask = (offs < elems_per_group) & (c_local < group_size)
        x_ptrs = X_ptr + n * C * spatial_size + c_global * spatial_size + s_idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / elems_per_group
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for off in range(0, elems_per_group, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        c_local = offs // spatial_size
        s_idx = offs % spatial_size
        c_global = g * group_size + c_local
        mask = (offs < elems_per_group) & (c_local < group_size)
        x_ptrs = X_ptr + n * C * spatial_size + c_global * spatial_size + s_idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_norm = (x - mean) * inv_std
        w = tl.load(W_ptr + c_global, mask=(c_global < C), other=1.0)
        b = tl.load(B_ptr + c_global, mask=(c_global < C), other=0.0)
        out = x_norm * w + b
        y_ptrs = Y_ptr + n * C * spatial_size + c_global * spatial_size + s_idx
        tl.store(y_ptrs, out, mask=mask)


def kernel_function(p_conv_transpose_weight, p_group_norm_weight, p_group_norm_bias, x):
    # x: (N, IN_C, D, H, W)
    # weight: (IN_C, OUT_C, KD, KH, KW)
    N, IN_C, IN_D, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KD, KH, KW = p_conv_transpose_weight.shape

    # ConvTranspose3d output size (stride=1, no padding, no output_padding)
    OUT_D = IN_D + KD - 1
    OUT_H = IN_H + KH - 1
    OUT_W_dim = IN_W + KW - 1

    M_TOTAL = N * OUT_D * OUT_H * OUT_W_dim
    TOTAL_K = IN_C * KD * KH * KW

    # Allocate output for conv+relu
    conv_out = torch.empty((N, OUT_C, OUT_D, OUT_H, OUT_W_dim), device=x.device, dtype=x.dtype)

    # Strides for x
    s_xn, s_xc, s_xd, s_xh, s_xw = x.stride()
    # Strides for weight (IN_C, OUT_C, KD, KH, KW)
    s_wi, s_wo, s_wkd, s_wkh, s_wkw = p_conv_transpose_weight.stride()
    # Strides for output
    s_yn, s_yc, s_yd, s_yh, s_yw = conv_out.stride()

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv_transpose3d_relu_kernel[grid](
        x, p_conv_transpose_weight, conv_out,
        N, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W_dim,
        KD, KH, KW,
        TOTAL_K,
        M_TOTAL,
        s_xn, s_xc, s_xd, s_xh, s_xw,
        s_wi, s_wo, s_wkd, s_wkh, s_wkw,
        s_yn, s_yc, s_yd, s_yh, s_yw,
    )

    # GroupNorm
    num_groups = 8
    group_size = OUT_C // num_groups
    spatial_size = OUT_D * OUT_H * OUT_W_dim
    eps = 1e-5

    gn_out = torch.empty_like(conv_out)

    total_programs = N * num_groups
    group_norm_kernel[(total_programs,)](
        conv_out, gn_out, p_group_norm_weight, p_group_norm_bias,
        N, OUT_C, spatial_size,
        num_groups, group_size,
        eps,
    )

    return gn_out


def get_inputs():
    p_conv_transpose_weight = torch.randn(64, 128, 3, 3, 3)
    p_group_norm_weight = torch.randn(128)
    p_group_norm_bias = torch.randn(128)
    x = torch.rand(16, 64, 32, 32, 32)
    return [p_conv_transpose_weight, p_group_norm_weight, p_group_norm_bias, x]


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
        weight = inputs[0].cuda()
        gn_w = inputs[1].cuda()
        gn_b = inputs[2].cuda()
        x_ref = inputs[3].cuda()

        conv_t = nn.ConvTranspose3d(64, 128, 3, bias=False).cuda()
        conv_t.weight.data = weight
        gn = nn.GroupNorm(8, 128).cuda()
        gn.weight.data = gn_w
        gn.bias.data = gn_b

        with torch.no_grad():
            ref = gn(torch.relu(conv_t(x_ref)))

        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("PASSED")
        else:
            print("FAILED")