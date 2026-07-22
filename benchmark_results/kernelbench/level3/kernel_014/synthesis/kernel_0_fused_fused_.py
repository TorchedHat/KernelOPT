import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU kernel (inference mode)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    # Total elements = N * C * HW
    # We index as (n, c, hw) where channel determines which mean/var/weight/bias to use
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * HW
    mask = offsets < total

    # Compute channel index
    hw_idx = offsets % HW
    c_idx = (offsets // HW) % C
    n_idx = offsets // (HW * C)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # BatchNorm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * w + b

    # ReLU
    out = tl.maximum(x_scaled, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# Implicit GEMM Conv2d kernel: fused with nothing (pure conv)
# Input: (N, C_in, H, W) in NCHW
# Weight: (C_out, C_in, KH, KW)
# Output: (N, C_out, H_out, W_out) in NCHW
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
    ],
    key=['M_total', 'C_out', 'C_in', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    # Strides for x (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # Strides for w (C_out, C_in, KH, KW)
    s_wco, s_wci, s_wkh, s_wkw,
    # Strides for out (NCHW)
    s_on, s_oc, s_oh, s_ow,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % W_out
    tmp = offs_m // W_out
    out_h_idx = tmp % H_out
    batch_idx = tmp // H_out

    mask_m = offs_m < M_total
    mask_n = offs_n < C_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = C_in * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K -> (c_in, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ci_idx = tmp_k // KH

        # Input coordinates
        ih = out_h_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw = out_w_idx[:, None] * stride_w + kw_idx[None, :] - pad_w

        valid_x = (mask_m[:, None] & mask_k[None, :] &
                   (ih >= 0) & (ih < H_in) &
                   (iw >= 0) & (iw < W_in) &
                   (batch_idx[:, None] < N))

        x_ptrs = (x_ptr +
                  batch_idx[:, None] * s_xn +
                  ci_idx[None, :] * s_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: (C_out, C_in, KH, KW)
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_ptrs = (w_ptr +
                  offs_n[None, :] * s_wco +
                  ci_idx[:, None] * s_wci +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output (NCHW)
    valid_out = mask_m[:, None] & mask_n[None, :]
    out_ptrs = (out_ptr +
                batch_idx[:, None] * s_on +
                offs_n[None, :] * s_oc +
                out_h_idx[:, None] * s_oh +
                out_w_idx[:, None] * s_ow)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def run_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total = N * C * H * W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        N, C, H * W,
        eps,
    )
    return out


def run_conv2d(x, w, stride=1, padding=1):
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = w.shape
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    M_total = N * H_out * W_out

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x, w, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride, stride,
        padding, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M_total,
    )
    return out


def kernel_function(
    p_layers_0_0_weight, p_layers_0_0_bias,
    p_layers_0_2_weight,
    p_layers_1_0_weight, p_layers_1_0_bias,
    p_layers_1_2_weight,
    p_layers_2_0_weight, p_layers_2_0_bias,
    p_layers_2_2_weight,
    p_layers_3_0_weight, p_layers_3_0_bias,
    p_layers_3_2_weight,
    p_layers_4_0_weight, p_layers_4_0_bias,
    p_layers_4_2_weight,
    p_layers_5_0_weight, p_layers_5_0_bias,
    p_layers_5_2_weight,
    b_layers_0_0_running_mean, b_layers_0_0_running_var,
    b_layers_1_0_running_mean, b_layers_1_0_running_var,
    b_layers_2_0_running_mean, b_layers_2_0_running_var,
    b_layers_3_0_running_mean, b_layers_3_0_running_var,
    b_layers_4_0_running_mean, b_layers_4_0_running_var,
    b_layers_5_0_running_mean, b_layers_5_0_running_var,
    x,
):
    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3) + Dropout(0.0) + Cat
    bn0 = run_bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
    conv0 = run_conv2d(bn0, p_layers_0_2_weight)
    # dropout(0.0) is identity in inference
    x1 = torch.cat([x, conv0], dim=1)  # (10, 64, 224, 224)

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3) + Cat
    bn1 = run_bn_relu(x1, b_layers_1_0_running_mean, b_layers_1_0_running_var,
    conv1 = run_conv2d(bn1, p_layers_1_2_weight)
    x2 = torch.cat([x1, conv1], dim=1)  # (10, 96, 224, 224)

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3) + Cat
    bn2 = run_bn_relu(x2, b_layers_2_0_running_mean, b_layers_2_0_running_var,
    conv2 = run_conv2d(bn2, p_layers_2_2_weight)
    x3 = torch.cat([x2, conv2], dim=1)  # (10, 128, 224, 224)

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3) + Cat
    bn3 = run_bn_relu(x3, b_layers_3_0_running_mean, b_layers_3_0_running_var,
    conv3 = run_conv2d(bn3, p_layers_3_2_weight)
    x4 = torch.cat([x3, conv3], dim=1)  # (10, 160, 224, 224)

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3) + Cat
    bn4 = run_bn_relu(x4, b_layers_4_0_running_mean, b_layers_4_0_running_var,
    conv4 = run_conv2d(bn4, p_layers_4_2_weight)
    x5 = torch.cat([x4, conv4], dim=1)  # (10, 192, 224, 224)

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3) + Cat
    bn5 = run_bn_relu(x5, b_layers_5_0_running_mean, b_layers_5_0_running_var,
    conv5 = run_conv2d(bn5, p_layers_5_2_weight)
    x6 = torch.cat([x5, conv5], dim=1)  # (10, 224, 224, 224)

    return x6


def get_inputs():
    torch.manual_seed(42)
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224

    p_layers_0_0_weight = torch.randn(32, device='cuda')
    p_layers_0_0_bias = torch.randn(32, device='cuda')
    p_layers_0_2_weight = torch.randn(32, 32, 3, 3, device='cuda') * 0.1

    p_layers_1_0_weight = torch.randn(64, device='cuda')
    p_layers_1_0_bias = torch.randn(64, device='cuda')
    p_layers_1_2_weight = torch.randn(32, 64, 3, 3, device='cuda') * 0.1

    p_layers_2_0_weight = torch.randn(96, device='cuda')
    p_layers_2_0_bias = torch.randn(96, device='cuda')
    p_layers_2_2_weight = torch.randn(32, 96, 3, 3, device='cuda') * 0.1

    p_layers_3_0_weight = torch.randn(128, device='cuda')
    p_layers_3_0_bias = torch.randn(128, device='cuda')
    p_layers_3_2_weight = torch.randn(32, 128, 3, 3, device='cuda') * 0.1

    p_layers_4_0_weight = torch.randn(160, device='cuda')
    p_layers_4_0_bias = torch.randn(160, device='cuda')
    p_layers_4_2_weight = torch.randn(32, 160, 3, 3, device='cuda') * 0.1

    p_layers_5_0_weight = torch.randn(192, device='cuda')
    p_layers_5_0_bias = torch.randn(192, device='cuda')
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

    x = torch.randn(batch_size, num_input_features, height, width, device='cuda')

    return [
        p_layers_0_0_weight, p_layers_0_0_bias,
        p_layers_0_2_weight,
        p_layers_1_0_weight, p_layers_1_0_bias,
        p_layers_1_2_weight,
        p_layers_2_0_weight, p_layers_2_0_bias,
        p_layers_2_2_weight,
        p_layers_3_0_weight, p_layers_3_0_bias,
        p_layers_3_2_weight,
        p_layers_4_0_weight, p_layers_4_0_bias,
        p_layers_4_2_weight,
        p_layers_5_0_weight, p_layers_5_0_bias,
        p_layers_5_2_weight,
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

        # Build reference model
        class Model(nn.Module):
            def __init__(self, num_layers, num_input_features, growth_rate):
                super(Model, self).__init__()
                layers = []
                for i in range(num_layers):
                    layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
                self.layers = nn.ModuleList(layers)

            def _make_layer(self, in_features, growth_rate):
                return nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.Dropout(0.0)
                )

            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_feature = layer(x)
                    features.append(new_feature)
                    x = torch.cat(features, 1)
                return x

        model = Model(6, 32, 32).cuda().eval()

        # Set weights
        (p_layers_0_0_weight, p_layers_0_0_bias,
         p_layers_0_2_weight,
         p_layers_1_0_weight, p_layers_1_0_bias,
         p_layers_1_2_weight,
         p_layers_2_0_weight, p_layers_2_0_bias,
         p_layers_2_2_weight,
         p_layers_3_0_weight, p_layers_3_0_bias,
         p_layers_3_2_weight,
         p_layers_4_0_weight, p_layers_4_0_bias,
         p_layers_4_2_weight,
         p_layers_5_0_weight, p_layers_5_0_bias,
         p_layers_5_2_weight,
         b_layers_0_0_running_mean, b_layers_0_0_running_var,
         b_layers_1_0_running_mean, b_layers_1_0_running_var,
         b_layers_2_0_running_mean, b_layers_2_0_running_var,
         b_layers_3_0_running_mean, b_layers_3_0_running_var,
         b_layers_4_0_running_mean, b_layers_4_0_running_var,
         b_layers_5_0_running_mean, b_layers_5_0_running_var,
         x) = inputs

        with torch.no_grad():
            model.layers[0][0].weight.data.copy_(p_layers_0_0_weight)
            model.layers[0][0].bias.data.copy_(p_layers_0_0_bias)
            model.layers[0][0].running_mean.data.copy_(b_layers_0_0_running_mean)
            model.layers[0][0].running_var.data.copy_(b_layers_0_0_running_var)
            model.layers[0][2].weight.data.copy_(p_layers_0_2_weight)

            model.layers[1][0].weight.data.copy_(p_layers_1_0_weight)
            model.layers[1][0].bias.data.copy_(p_layers_1_0_bias)
            model.layers[1][0].running_mean.data.copy_(b_layers_1_0_running_mean)
            model.layers[1][0].running_var.data.copy_(b_layers_1_0_running_var)
            model.layers[1][2].weight.data.copy_(p_layers_1_2_weight)

            model.layers[2][0].weight.data.copy_(p_layers_2_0_weight)
            model.layers[2][0].bias.data.copy_(p_layers_2_0_bias)
            model.layers[2][0].running_mean.data.copy_(b_layers_2_0_running_mean)
            model.layers[2][0].running_var.data.copy_(b_layers_2_0_running_var)
            model.layers[2][2].weight.data.copy_(p_layers_2_2_weight)

            model.layers[3][0].weight.data.copy_(p_layers_3_0_weight)
            model.layers[3][0].bias.data.copy_(p_layers_3_0_bias)
            model.layers[3][0].running_mean.data.copy_(b_layers_3_0_running_mean)
            model.layers[3][0].running_var.data.copy_(b_layers_3_0_running_var)
            model.layers[3][2].weight.data.copy_(p_layers_3_2_weight)

            model.layers[4][0].weight.data.copy_(p_layers_4_0_weight)
            model.layers[4][0].bias.data.copy_(p_layers_4_0_bias)
            model.layers[4][0].running_mean.data.copy_(b_layers_4_0_running_mean)
            model.layers[4][0].running_var.data.copy_(b_layers_4_0_running_var)
            model.layers[4][2].weight.data.copy_(p_layers_4_2_weight)

            model.layers[5][0].weight.data.copy_(p_layers_5_0_weight)
            model.layers[5][0].bias.data.copy_(p_layers_5_0_bias)
            model.layers[5][0].running_mean.data.copy_(b_layers_5_0_running_mean)
            model.layers[5][0].running_var.data.copy_(b_layers_5_0_running_var)
            model.layers[5][2].weight.data.copy_(p_layers_5_2_weight)

            ref = model(x)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")