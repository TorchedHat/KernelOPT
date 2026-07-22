import torch
import triton
import triton.language as tl

# Fused BatchNorm (inference) + ReLU kernel
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
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    # Total elements = N * C * HW
    # Channel index = (flat_idx // HW) % C
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * HW
    mask = offsets < total

    # Compute channel index
    hw_idx = offsets % HW
    c_idx = (offsets // HW) % C
    n_idx = offsets // (HW * C)

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load per-channel stats
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * weight + bias

    # ReLU
    out = tl.maximum(x_norm, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    total = N * C * HW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        N, C, HW,
        eps,
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

    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3, pad=1) + cat
    bn0 = fused_bn_relu(x, p_layers_0_0_weight, p_layers_0_0_bias,
    # Conv2d: input=bn0 (10,32,224,224), weight=(32,32,3,3), no bias, pad=1
    conv0 = torch.nn.functional.conv2d(bn0, p_layers_0_2_weight, bias=None, stride=1, padding=1)
    # dropout(0.0) is identity
    x = torch.cat(features + [conv0], dim=1)  # (10, 64, 224, 224)
    features.append(conv0)

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3, pad=1) + cat
    bn1 = fused_bn_relu(x, p_layers_1_0_weight, p_layers_1_0_bias,
    conv1 = torch.nn.functional.conv2d(bn1, p_layers_1_2_weight, bias=None, stride=1, padding=1)
    x = torch.cat(features + [conv1], dim=1)  # (10, 96, 224, 224)
    features.append(conv1)

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3, pad=1) + cat
    bn2 = fused_bn_relu(x, p_layers_2_0_weight, p_layers_2_0_bias,
    conv2 = torch.nn.functional.conv2d(bn2, p_layers_2_2_weight, bias=None, stride=1, padding=1)
    x = torch.cat(features + [conv2], dim=1)  # (10, 128, 224, 224)
    features.append(conv2)

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3, pad=1) + cat
    bn3 = fused_bn_relu(x, p_layers_3_0_weight, p_layers_3_0_bias,
    conv3 = torch.nn.functional.conv2d(bn3, p_layers_3_2_weight, bias=None, stride=1, padding=1)
    x = torch.cat(features + [conv3], dim=1)  # (10, 160, 224, 224)
    features.append(conv3)

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3, pad=1) + cat
    bn4 = fused_bn_relu(x, p_layers_4_0_weight, p_layers_4_0_bias,
    conv4 = torch.nn.functional.conv2d(bn4, p_layers_4_2_weight, bias=None, stride=1, padding=1)
    x = torch.cat(features + [conv4], dim=1)  # (10, 192, 224, 224)
    features.append(conv4)

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3, pad=1) + cat
    bn5 = fused_bn_relu(x, p_layers_5_0_weight, p_layers_5_0_bias,
    conv5 = torch.nn.functional.conv2d(bn5, p_layers_5_2_weight, bias=None, stride=1, padding=1)
    x = torch.cat(features + [conv5], dim=1)  # (10, 224, 224, 224)

    return x


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    device = 'cuda'

    p_layers_0_0_weight = torch.ones(32, device=device)
    p_layers_0_0_bias = torch.zeros(32, device=device)
    p_layers_0_2_weight = torch.randn(32, 32, 3, 3, device=device) * 0.1

    p_layers_1_0_weight = torch.ones(64, device=device)
    p_layers_1_0_bias = torch.zeros(64, device=device)
    p_layers_1_2_weight = torch.randn(32, 64, 3, 3, device=device) * 0.1

    p_layers_2_0_weight = torch.ones(96, device=device)
    p_layers_2_0_bias = torch.zeros(96, device=device)
    p_layers_2_2_weight = torch.randn(32, 96, 3, 3, device=device) * 0.1

    p_layers_3_0_weight = torch.ones(128, device=device)
    p_layers_3_0_bias = torch.zeros(128, device=device)
    p_layers_3_2_weight = torch.randn(32, 128, 3, 3, device=device) * 0.1

    p_layers_4_0_weight = torch.ones(160, device=device)
    p_layers_4_0_bias = torch.zeros(160, device=device)
    p_layers_4_2_weight = torch.randn(32, 160, 3, 3, device=device) * 0.1

    p_layers_5_0_weight = torch.ones(192, device=device)
    p_layers_5_0_bias = torch.zeros(192, device=device)
    p_layers_5_2_weight = torch.randn(32, 192, 3, 3, device=device) * 0.1

    b_layers_0_0_running_mean = torch.zeros(32, device=device)
    b_layers_0_0_running_var = torch.ones(32, device=device)
    b_layers_1_0_running_mean = torch.zeros(64, device=device)
    b_layers_1_0_running_var = torch.ones(64, device=device)
    b_layers_2_0_running_mean = torch.zeros(96, device=device)
    b_layers_2_0_running_var = torch.ones(96, device=device)
    b_layers_3_0_running_mean = torch.zeros(128, device=device)
    b_layers_3_0_running_var = torch.ones(128, device=device)
    b_layers_4_0_running_mean = torch.zeros(160, device=device)
    b_layers_4_0_running_var = torch.ones(160, device=device)
    b_layers_5_0_running_mean = torch.zeros(192, device=device)
    b_layers_5_0_running_var = torch.ones(192, device=device)

    x = torch.rand(batch_size, num_input_features, height, width, device=device)

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
        device = 'cuda'
        num_layers = 6
        num_input_features = 32
        growth_rate = 32

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

        def make_layer(in_features, growth_rate):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
                nn.Dropout(0.0)
            )
            return layer

        layers_params = [
            (p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
             b_layers_0_0_running_mean, b_layers_0_0_running_var, 32),
            (p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
             b_layers_1_0_running_mean, b_layers_1_0_running_var, 64),
            (p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
             b_layers_2_0_running_mean, b_layers_2_0_running_var, 96),
            (p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
             b_layers_3_0_running_mean, b_layers_3_0_running_var, 128),
            (p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
             b_layers_4_0_running_mean, b_layers_4_0_running_var, 160),
            (p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
             b_layers_5_0_running_mean, b_layers_5_0_running_var, 192),
        ]

        import torch.nn.functional as F

        def ref_forward(x_in):
            features = [x_in]
            cur_x = x_in
            for (w, b, conv_w, rm, rv, in_c) in layers_params:
                # BN + ReLU on cur_x
                bn_out = F.batch_norm(cur_x, rm, rv, w, b, training=False, eps=1e-5)
                relu_out = F.relu(bn_out, inplace=False)
                conv_out = F.conv2d(relu_out, conv_w, bias=None, stride=1, padding=1)
                features.append(conv_out)
                cur_x = torch.cat(features, dim=1)
            return cur_x

        ref = ref_forward(x)
        torch.cuda.synchronize()

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")