import torch
import triton
import triton.language as tl

# Fused BN+ReLU pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C, HW,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # offsets index into (N, C, H, W) flattened as N*C*HW
    # channel index
    c = (offsets // HW) % C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# Implicit GEMM Conv2d kernel: BN+ReLU fused as pre-processing done separately
# Conv2d: input (N, Ci, H, W) NCHW, weight (Co, Ci, KH, KW)
# Output: (N, Co, OH, OW)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    # strides for X (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # strides for W (Co, Ci, KH, KW)
    s_wo, s_wi, s_wkh, s_wkw,
    # strides for Out (NCHW)
    s_on, s_oc, s_oh, s_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid_x = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & valid_m[:, None] & valid_k[None, :]

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * s_wo + in_c[:, None] * s_wi + kh[:, None] * s_wkh + kw[:, None] * s_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output
    valid_out_m = offs_m < M_TOTAL
    valid_out_n = offs_n < OUT_C
    out_ptrs = Out_ptr + batch[:, None] * s_on + offs_n[None, :] * s_oc + out_h[:, None] * s_oh + out_w[:, None] * s_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out_m[:, None] & valid_out_n[None, :])


def run_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = N * C * H * W
    HW = H * W
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        N, C, HW,
        eps,
        n_elements,
    )
    return out


def run_conv2d(x, w, stride=1, padding=1):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    STRIDE_H = STRIDE_W = stride
    PAD_H = PAD_W = padding
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wo, s_wi, s_wkh, s_wkw = w.stride()
    s_on, s_oc, s_oh, s_ow = out.stride()

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, w, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M_TOTAL, TOTAL_K,
        s_xn, s_xc, s_xh, s_xw,
        s_wo, s_wi, s_wkh, s_wkw,
        s_on, s_oc, s_oh, s_ow,
    )
    return out


def kernel_function(
    p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
    p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
    p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
    p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
    p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
    p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
    b_layers_0_0_running_mean, b_layers_0_0_running_var,
    b_layers_1_0_running_mean, b_layers_1_0_running_var,
    b_layers_2_0_running_mean, b_layers_2_0_running_var,
    b_layers_3_0_running_mean, b_layers_3_0_running_var,
    b_layers_4_0_running_mean, b_layers_4_0_running_var,
    b_layers_5_0_running_mean, b_layers_5_0_running_var,
    x,
):
    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3) + Cat -> [10, 64, 224, 224]
    bn0 = run_bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
    conv0 = run_conv2d(bn0, p_layers_0_2_weight)
    x1 = torch.cat([x, conv0], dim=1)  # [10, 64, 224, 224]

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3) + Cat -> [10, 96, 224, 224]
    bn1 = run_bn_relu(x1, b_layers_1_0_running_mean, b_layers_1_0_running_var,
    conv1 = run_conv2d(bn1, p_layers_1_2_weight)
    x2 = torch.cat([x1, conv1], dim=1)  # [10, 96, 224, 224]

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3) + Cat -> [10, 128, 224, 224]
    bn2 = run_bn_relu(x2, b_layers_2_0_running_mean, b_layers_2_0_running_var,
    conv2 = run_conv2d(bn2, p_layers_2_2_weight)
    x3 = torch.cat([x2, conv2], dim=1)  # [10, 128, 224, 224]

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3) + Cat -> [10, 160, 224, 224]
    bn3 = run_bn_relu(x3, b_layers_3_0_running_mean, b_layers_3_0_running_var,
    conv3 = run_conv2d(bn3, p_layers_3_2_weight)
    x4 = torch.cat([x3, conv3], dim=1)  # [10, 160, 224, 224]

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3) + Cat -> [10, 192, 224, 224]
    bn4 = run_bn_relu(x4, b_layers_4_0_running_mean, b_layers_4_0_running_var,
    conv4 = run_conv2d(bn4, p_layers_4_2_weight)
    x5 = torch.cat([x4, conv4], dim=1)  # [10, 192, 224, 224]

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3) + Cat -> [10, 224, 224, 224]
    bn5 = run_bn_relu(x5, b_layers_5_0_running_mean, b_layers_5_0_running_var,
    conv5 = run_conv2d(bn5, p_layers_5_2_weight)
    x6 = torch.cat([x5, conv5], dim=1)  # [10, 224, 224, 224]

    return x6


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    num_layers = 6

    p_layers_0_0_weight = torch.ones(32, device='cuda')
    p_layers_0_0_bias = torch.zeros(32, device='cuda')
    p_layers_0_2_weight = torch.randn(32, 32, 3, 3, device='cuda') * 0.1

    p_layers_1_0_weight = torch.ones(64, device='cuda')
    p_layers_1_0_bias = torch.zeros(64, device='cuda')
    p_layers_1_2_weight = torch.randn(32, 64, 3, 3, device='cuda') * 0.1

    p_layers_2_0_weight = torch.ones(96, device='cuda')
    p_layers_2_0_bias = torch.zeros(96, device='cuda')
    p_layers_2_2_weight = torch.randn(32, 96, 3, 3, device='cuda') * 0.1

    p_layers_3_0_weight = torch.ones(128, device='cuda')
    p_layers_3_0_bias = torch.zeros(128, device='cuda')
    p_layers_3_2_weight = torch.randn(32, 128, 3, 3, device='cuda') * 0.1

    p_layers_4_0_weight = torch.ones(160, device='cuda')
    p_layers_4_0_bias = torch.zeros(160, device='cuda')
    p_layers_4_2_weight = torch.randn(32, 160, 3, 3, device='cuda') * 0.1

    p_layers_5_0_weight = torch.ones(192, device='cuda')
    p_layers_5_0_bias = torch.zeros(192, device='cuda')
    p_layers_5_2_weight = torch.randn(32, 192, 3, 3, device='cuda') * 0.1

    b_layers_0_0_running_mean = torch.zeros(32, device='cuda')
    b_layers_0_0_running_var = torch.ones(32, device='cuda')
    b_layers_1_0_running_mean = torch.zeros(64, device='cuda')
    b_layers_1_0_running_var = torch.ones(64, device='cuda')
    b_layers_2_0_running_mean = torch.zeros(96, device='cuda')
    b_layers_2_0_running_var = torch.ones(96, device='cuda')
    b_layers_3_0_running_mean = torch.zeros(128, device='cuda')
    b_layers_3_0_running_var = torch.ones(128, device='cuda')
    b_layers_4_0_running_mean = torch.zeros(160, device='cuda')
    b_layers_4_0_running_var = torch.ones(160, device='cuda')
    b_layers_5_0_running_mean = torch.zeros(192, device='cuda')
    b_layers_5_0_running_var = torch.ones(192, device='cuda')

    x = torch.rand(batch_size, num_input_features, height, width, device='cuda')

    return [
        p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
        p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
        p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
        p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
        p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
        p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
        b_layers_0_0_running_mean, b_layers_0_0_running_var,
        b_layers_1_0_running_mean, b_layers_1_0_running_var,
        b_layers_2_0_running_mean, b_layers_2_0_running_var,
        b_layers_3_0_running_mean, b_layers_3_0_running_var,
        b_layers_4_0_running_mean, b_layers_4_0_running_var,
        b_layers_5_0_running_mean, b_layers_5_0_running_var,
        x,
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
        import torch.nn as nn
        import torch.nn.functional as F

        (p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
         p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
         p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
         p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
         p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
         p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
         b_layers_0_0_running_mean, b_layers_0_0_running_var,
         b_layers_1_0_running_mean, b_layers_1_0_running_var,
         b_layers_2_0_running_mean, b_layers_2_0_running_var,
         b_layers_3_0_running_mean, b_layers_3_0_running_var,
         b_layers_4_0_running_mean, b_layers_4_0_running_var,
         b_layers_5_0_running_mean, b_layers_5_0_running_var,
         x) = inputs

        def ref_bn_relu(inp, mean, var, weight, bias, eps=1e-5):
            return F.relu(F.batch_norm(inp, mean, var, weight, bias, training=False, eps=eps))

        def ref_conv(inp, w):
            return F.conv2d(inp, w, bias=None, stride=1, padding=1)

        x0 = x
        bn0 = ref_bn_relu(x0, b_layers_0_0_running_mean, b_layers_0_0_running_var,
        c0 = ref_conv(bn0, p_layers_0_2_weight)
        x1 = torch.cat([x0, c0], dim=1)

        bn1 = ref_bn_relu(x1, b_layers_1_0_running_mean, b_layers_1_0_running_var,
        c1 = ref_conv(bn1, p_layers_1_2_weight)
        x2 = torch.cat([x1, c1], dim=1)

        bn2 = ref_bn_relu(x2, b_layers_2_0_running_mean, b_layers_2_0_running_var,
        c2 = ref_conv(bn2, p_layers_2_2_weight)
        x3 = torch.cat([x2, c2], dim=1)

        bn3 = ref_bn_relu(x3, b_layers_3_0_running_mean, b_layers_3_0_running_var,
        c3 = ref_conv(bn3, p_layers_3_2_weight)
        x4 = torch.cat([x3, c3], dim=1)

        bn4 = ref_bn_relu(x4, b_layers_4_0_running_mean, b_layers_4_0_running_var,
        c4 = ref_conv(bn4, p_layers_4_2_weight)
        x5 = torch.cat([x4, c4], dim=1)

        bn5 = ref_bn_relu(x5, b_layers_5_0_running_mean, b_layers_5_0_running_var,
        c5 = ref_conv(bn5, p_layers_5_2_weight)
        ref = torch.cat([x5, c5], dim=1)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")