import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['IN_C', 'M_TOTAL'],
)
@triton.jit
def bn_relu_conv2d_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    Out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M_TOTAL,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
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

    TOTAL_K = IN_C * 9  # 3x3 kernel

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % 3
        kh = offs_k // (IN_C * 3)

        ih = out_h[:, None] * 1 + kh[None, :] - 1  # stride=1, pad=1
        iw = out_w[:, None] * 1 + kw[None, :] - 1

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_b = batch[:, None] < BATCH
        mask_x = valid_m & valid_k & valid_h & valid_w & valid_b

        x_ptrs = X_ptr + (
            batch[:, None] * stride_xn +
            in_c[None, :] * stride_xc +
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Apply BN+ReLU to x_tile
        # Load per-channel BN params
        in_c_clamped = tl.where(valid_k, in_c, 0)
        mean_val = tl.load(mean_ptr + in_c_clamped[None, :], mask=valid_k, other=0.0)
        var_val = tl.load(var_ptr + in_c_clamped[None, :], mask=valid_k, other=1.0)
        gamma_val = tl.load(gamma_ptr + in_c_clamped[None, :], mask=valid_k, other=1.0)
        beta_val = tl.load(beta_ptr + in_c_clamped[None, :], mask=valid_k, other=0.0)

        x_norm = (x_tile - mean_val) / tl.sqrt(var_val + 1e-5)
        x_bn = x_norm * gamma_val + beta_val
        x_relu = tl.where(x_bn > 0, x_bn, tl.zeros_like(x_bn))
        x_relu = tl.where(mask_x, x_relu, tl.zeros_like(x_relu))

        # Load weight tile [BLOCK_K, BLOCK_N]
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n

        w_ptrs = W_ptr + (
            offs_n[None, :] * stride_wn +
            in_c[:, None] * stride_wc +
            kh[:, None] * stride_wh +
            kw[:, None] * stride_ww
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_relu, w_tile, acc)

    # Store output
    valid_m_out = offs_m[:, None] < M_TOTAL
    valid_n_out = offs_n[None, :] < OUT_C
    valid_b_out = batch[:, None] < BATCH
    mask_out = valid_m_out & valid_n_out & valid_b_out

    out_ptrs = Out_ptr + (
        batch[:, None] * stride_on +
        offs_n[None, :] * stride_oc +
        out_h[:, None] * stride_oh +
        out_w[:, None] * stride_ow
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def bn_relu_conv2d(x, weight, mean, var, gamma, beta):
    """
    Fused BatchNorm + ReLU + Conv2d(3x3, pad=1, stride=1)
    """
    N, IN_C, H, W = x.shape
    OUT_C = weight.shape[0]
    OUT_H, OUT_W = H, W  # same spatial size with pad=1, stride=1

    M_TOTAL = N * OUT_H * OUT_W
    out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Strides for x (NCHW)
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    # Strides for weight (OUT_C, IN_C, KH, KW)
    stride_wn, stride_wc, stride_wh, stride_ww = weight.stride()
    # Strides for output (NCHW)
    stride_on, stride_oc, stride_oh, stride_ow = out.stride()

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    bn_relu_conv2d_kernel[grid](
        x, weight,
        mean, var, gamma, beta,
        out,
        N, IN_C, H, W,
        OUT_C, OUT_H, OUT_W,
        M_TOTAL,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wh, stride_ww,
        stride_on, stride_oc, stride_oh, stride_ow,
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
    features = [x]

    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3)
    new_feat = bn_relu_conv2d(
        x,
        p_layers_0_2_weight,
        b_layers_0_0_running_mean, b_layers_0_0_running_var,
        p_layers_0_0_weight, p_layers_0_0_bias,
    )
    features.append(new_feat)
    x = torch.cat(features, dim=1)  # (10, 64, 224, 224)

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3)
    new_feat = bn_relu_conv2d(
        x,
        p_layers_1_2_weight,
        b_layers_1_0_running_mean, b_layers_1_0_running_var,
        p_layers_1_0_weight, p_layers_1_0_bias,
    )
    features.append(new_feat)
    x = torch.cat(features, dim=1)  # (10, 96, 224, 224)

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3)
    new_feat = bn_relu_conv2d(
        x,
        p_layers_2_2_weight,
        b_layers_2_0_running_mean, b_layers_2_0_running_var,
        p_layers_2_0_weight, p_layers_2_0_bias,
    )
    features.append(new_feat)
    x = torch.cat(features, dim=1)  # (10, 128, 224, 224)

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3)
    new_feat = bn_relu_conv2d(
        x,
        p_layers_3_2_weight,
        b_layers_3_0_running_mean, b_layers_3_0_running_var,
        p_layers_3_0_weight, p_layers_3_0_bias,
    )
    features.append(new_feat)
    x = torch.cat(features, dim=1)  # (10, 160, 224, 224)

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3)
    new_feat = bn_relu_conv2d(
        x,
        p_layers_4_2_weight,
        b_layers_4_0_running_mean, b_layers_4_0_running_var,
        p_layers_4_0_weight, p_layers_4_0_bias,
    )
    features.append(new_feat)
    x = torch.cat(features, dim=1)  # (10, 192, 224, 224)

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3)
    new_feat = bn_relu_conv2d(
        x,
        p_layers_5_2_weight,
        b_layers_5_0_running_mean, b_layers_5_0_running_var,
        p_layers_5_0_weight, p_layers_5_0_bias,
    )
    features.append(new_feat)
    x = torch.cat(features, dim=1)  # (10, 224, 224, 224)

    return x


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224

    p_layers_0_0_weight = torch.randn(32, device='cuda')
    p_layers_0_0_bias = torch.randn(32, device='cuda')
    p_layers_0_2_weight = torch.randn(32, 32, 3, 3, device='cuda')

    p_layers_1_0_weight = torch.randn(64, device='cuda')
    p_layers_1_0_bias = torch.randn(64, device='cuda')
    p_layers_1_2_weight = torch.randn(32, 64, 3, 3, device='cuda')

    p_layers_2_0_weight = torch.randn(96, device='cuda')
    p_layers_2_0_bias = torch.randn(96, device='cuda')
    p_layers_2_2_weight = torch.randn(32, 96, 3, 3, device='cuda')

    p_layers_3_0_weight = torch.randn(128, device='cuda')
    p_layers_3_0_bias = torch.randn(128, device='cuda')
    p_layers_3_2_weight = torch.randn(32, 128, 3, 3, device='cuda')

    p_layers_4_0_weight = torch.randn(160, device='cuda')
    p_layers_4_0_bias = torch.randn(160, device='cuda')
    p_layers_4_2_weight = torch.randn(32, 160, 3, 3, device='cuda')

    p_layers_5_0_weight = torch.randn(192, device='cuda')
    p_layers_5_0_bias = torch.randn(192, device='cuda')
    p_layers_5_2_weight = torch.randn(32, 192, 3, 3, device='cuda')

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

        # Build reference model
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                num_input_features = 32
                growth_rate = 32
                for i in range(6):
                    in_f = num_input_features + i * growth_rate
                    layers.append(nn.Sequential(
                        nn.BatchNorm2d(in_f),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_f, growth_rate, 3, padding=1, bias=False),
                        nn.Dropout(0.0),
                    ))
                self.layers = nn.ModuleList(layers)

            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_feature = layer(x)
                    features.append(new_feature)
                    x = torch.cat(features, 1)
                return x

        model = Model().cuda().eval()

        # Copy weights
        x_in = inputs[-1]
        with torch.no_grad():
            # Set BN params
            for i, (lw, lb, rm, rv) in enumerate([
                (inputs[0], inputs[1], inputs[18], inputs[19]),
                (inputs[3], inputs[4], inputs[20], inputs[21]),
                (inputs[6], inputs[7], inputs[22], inputs[23]),
                (inputs[9], inputs[10], inputs[24], inputs[25]),
                (inputs[12], inputs[13], inputs[26], inputs[27]),
                (inputs[15], inputs[16], inputs[28], inputs[29]),
            ]):
                model.layers[i][0].weight.copy_(lw)
                model.layers[i][0].bias.copy_(lb)
                model.layers[i][0].running_mean.copy_(rm)
                model.layers[i][0].running_var.copy_(rv)

            # Set conv weights
            for i, cw in enumerate([inputs[2], inputs[5], inputs[8], inputs[11], inputs[14], inputs[17]]):
                model.layers[i][2].weight.copy_(cw)

        ref = model(x_in)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")