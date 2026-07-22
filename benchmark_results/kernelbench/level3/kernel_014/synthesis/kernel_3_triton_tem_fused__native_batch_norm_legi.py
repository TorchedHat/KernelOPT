import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_ELEMENTS'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    eps,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # offsets layout: [N, C, H, W] flattened
    # channel index = (offset // HW) % C
    hw_idx = offsets % HW
    nc_idx = offsets // HW
    c_idx = nc_idx % C

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


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = N * C * H * W
    HW = H * W

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        C, HW,
        eps,
        N_ELEMENTS=n_elements,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_ELEMENTS'],
)
@triton.jit
def cat_kernel(
    # We'll handle cat via a copy kernel that writes into the right slice
    src_ptr, dst_ptr,
    N, C_src, C_dst, HW,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # src layout: [N, C_src, H, W]
    # dst layout: [N, C_dst, H, W]
    # src element i -> batch b, channel c, spatial hw
    hw = offsets % HW
    nc = offsets // HW
    c = nc % C_src
    b = nc // C_src

    src_val = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    dst_idx = b * C_dst * HW + (src_offset + c) * HW + hw
    tl.store(dst_ptr + dst_idx, src_val, mask=mask)


def run_conv2d(x_bn_relu, weight):
    # Use PyTorch's cuDNN conv - fastest option
    import torch.nn.functional as F
    return F.conv2d(x_bn_relu, weight, bias=None, stride=1, padding=1)


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
):
    import torch.nn.functional as F

    # Layer 0: BN(32) + ReLU + Conv(32->32) + Cat -> [10, 64, 224, 224]
    x0_bn = bn_relu(x, p_layers_0_0_weight, p_layers_0_0_bias,
    x0_conv = F.conv2d(x0_bn, p_layers_0_2_weight, bias=None, stride=1, padding=1)
    # dropout(0.0) is identity
    x1 = torch.cat([x, x0_conv], dim=1)  # [10, 64, 224, 224]

    # Layer 1: BN(64) + ReLU + Conv(64->32) + Cat -> [10, 96, 224, 224]
    x1_bn = bn_relu(x1, p_layers_1_0_weight, p_layers_1_0_bias,
    x1_conv = F.conv2d(x1_bn, p_layers_1_2_weight, bias=None, stride=1, padding=1)
    x2 = torch.cat([x1, x1_conv], dim=1)  # [10, 96, 224, 224]

    # Layer 2: BN(96) + ReLU + Conv(96->32) + Cat -> [10, 128, 224, 224]
    x2_bn = bn_relu(x2, p_layers_2_0_weight, p_layers_2_0_bias,
    x2_conv = F.conv2d(x2_bn, p_layers_2_2_weight, bias=None, stride=1, padding=1)
    x3 = torch.cat([x2, x2_conv], dim=1)  # [10, 128, 224, 224]

    # Layer 3: BN(128) + ReLU + Conv(128->32) + Cat -> [10, 160, 224, 224]
    x3_bn = bn_relu(x3, p_layers_3_0_weight, p_layers_3_0_bias,
    x3_conv = F.conv2d(x3_bn, p_layers_3_2_weight, bias=None, stride=1, padding=1)
    x4 = torch.cat([x3, x3_conv], dim=1)  # [10, 160, 224, 224]

    # Layer 4: BN(160) + ReLU + Conv(160->32) + Cat -> [10, 192, 224, 224]
    x4_bn = bn_relu(x4, p_layers_4_0_weight, p_layers_4_0_bias,
    x4_conv = F.conv2d(x4_bn, p_layers_4_2_weight, bias=None, stride=1, padding=1)
    x5 = torch.cat([x4, x4_conv], dim=1)  # [10, 192, 224, 224]

    # Layer 5: BN(192) + ReLU + Conv(192->32) + Cat -> [10, 224, 224, 224]
    x5_bn = bn_relu(x5, p_layers_5_0_weight, p_layers_5_0_bias,
    x5_conv = F.conv2d(x5_bn, p_layers_5_2_weight, bias=None, stride=1, padding=1)
    x6 = torch.cat([x5, x5_conv], dim=1)  # [10, 224, 224, 224]

    return x6


def get_inputs():
    torch.manual_seed(42)
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    num_layers = 6

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

        # Load weights from inputs
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

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")