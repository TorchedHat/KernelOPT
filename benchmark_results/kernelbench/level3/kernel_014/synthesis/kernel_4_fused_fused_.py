import torch
import triton
import triton.language as tl

# BN+ReLU fused pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    numel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # offs layout: [N, C, H, W] -> channel = (offs // HW) % C
    c = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = tl.maximum(x_norm * w + b, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# Implicit GEMM Conv2d kernel: BN+ReLU already applied to input
# Input: (N, C_in, H, W) contiguous
# Weight: (C_out, C_in, KH, KW) contiguous
# Output: (N, C_out, H, W) contiguous
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, H, W, C_out,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    OH, OW,
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    batch = offs_m // (OW * OH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (c_in, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        c_in = offs_k // (KH * KW)

        # Input coordinates
        ih = oh[:, None] * stride_h + kh[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] - pad_w

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = (
            valid_m[:, None] &
            valid_k[None, :] &
            (ih >= 0) & (ih < H) &
            (iw >= 0) & (iw < W) &
            (batch[:, None] < N)
        )

        x_idx = batch[:, None] * (C_in * H * W) + c_in[None, :] * (H * W) + ih * W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

        # Weight: (C_out, C_in, KH, KW) -> index [offs_n, c_in, kh, kw]
        valid_n = offs_n < C_out
        w_idx = offs_n[None, :] * (C_in * KH * KW) + c_in[:, None] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        w_mask = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output
    valid_m = offs_m < M
    valid_n = offs_n < C_out
    out_idx = batch[:, None] * (C_out * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=valid_m[:, None] & valid_n[None, :])


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        C, HW,
        numel,
        eps,
    )
    return out


def conv2d_triton(x, w, pad=1, stride=1):
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = w.shape
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W_in + 2 * pad - KW) // stride + 1
    out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    M = N * OH * OW
    K = C_in * KH * KW
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv2d_implicit_gemm_kernel[grid](
        x, w, out,
        N, C_in, H, W_in, C_out,
        KH, KW,
        stride, stride,
        pad, pad,
        OH, OW,
        M, C_out, K,
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
    features = [x]

    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3)
    bn0 = bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
    conv0 = conv2d_triton(bn0, p_layers_0_2_weight)
    features.append(conv0)
    x = torch.cat(features, dim=1)

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3)
    bn1 = bn_relu(x, b_layers_1_0_running_mean, b_layers_1_0_running_var,
    conv1 = conv2d_triton(bn1, p_layers_1_2_weight)
    features.append(conv1)
    x = torch.cat(features, dim=1)

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3)
    bn2 = bn_relu(x, b_layers_2_0_running_mean, b_layers_2_0_running_var,
    conv2 = conv2d_triton(bn2, p_layers_2_2_weight)
    features.append(conv2)
    x = torch.cat(features, dim=1)

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3)
    bn3 = bn_relu(x, b_layers_3_0_running_mean, b_layers_3_0_running_var,
    conv3 = conv2d_triton(bn3, p_layers_3_2_weight)
    features.append(conv3)
    x = torch.cat(features, dim=1)

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3)
    bn4 = bn_relu(x, b_layers_4_0_running_mean, b_layers_4_0_running_var,
    conv4 = conv2d_triton(bn4, p_layers_4_2_weight)
    features.append(conv4)
    x = torch.cat(features, dim=1)

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3)
    bn5 = bn_relu(x, b_layers_5_0_running_mean, b_layers_5_0_running_var,
    conv5 = conv2d_triton(bn5, p_layers_5_2_weight)
    features.append(conv5)
    x = torch.cat(features, dim=1)

    return x


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    num_layers = 6

    inputs = []
    # BN weights/biases for each layer
    in_feats = num_input_features
    layer_params = []
    for i in range(num_layers):
        layer_params.append((in_feats, growth_rate))
        in_feats += growth_rate

    # p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight
    for i, (in_f, gr) in enumerate(layer_params):
        inputs.append(torch.randn(in_f, device='cuda'))  # BN weight
        inputs.append(torch.randn(in_f, device='cuda'))  # BN bias
        inputs.append(torch.randn(gr, in_f, 3, 3, device='cuda'))  # Conv weight

    # running_mean, running_var for each layer
    for i, (in_f, gr) in enumerate(layer_params):
        inputs.append(torch.zeros(in_f, device='cuda'))  # running_mean
        inputs.append(torch.ones(in_f, device='cuda'))   # running_var

    # x
    inputs.append(torch.randn(batch_size, num_input_features, height, width, device='cuda'))

    return inputs


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
            def __init__(self, num_layers, num_input_features, growth_rate):
                super().__init__()
                layers = []
                for i in range(num_layers):
                    in_f = num_input_features + i * growth_rate
                    layers.append(nn.Sequential(
                        nn.BatchNorm2d(in_f),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_f, growth_rate, kernel_size=3, padding=1, bias=False),
                        nn.Dropout(0.0)
                    ))
                self.layers = nn.ModuleList(layers)

            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_feature = layer(x)
                    features.append(new_feature)
                    x = torch.cat(features, 1)
                return x

        model = Model(6, 32, 32).cuda().eval()

        # Extract parameters from inputs
        # inputs order: [bn_w0, bn_b0, conv_w0, bn_w1, bn_b1, conv_w1, ..., rm0, rv0, rm1, rv1, ..., x]
        layer_params_list = [(32, 32), (64, 32), (96, 32), (128, 32), (160, 32), (192, 32)]
        idx = 0
        for i, (in_f, gr) in enumerate(layer_params_list):
            model.layers[i][0].weight.data = inputs[idx].clone()
            model.layers[i][0].bias.data = inputs[idx+1].clone()
            model.layers[i][2].weight.data = inputs[idx+2].clone()
            idx += 3

        for i, (in_f, gr) in enumerate(layer_params_list):
            model.layers[i][0].running_mean.data = inputs[idx].clone()
            model.layers[i][0].running_var.data = inputs[idx+1].clone()
            idx += 2

        x_ref = inputs[idx].clone()

        with torch.no_grad():
            ref = model(x_ref)

        print(f"Output shape: {output.shape}, ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")