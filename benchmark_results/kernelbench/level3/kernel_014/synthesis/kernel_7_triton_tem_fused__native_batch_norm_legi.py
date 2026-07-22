import torch
import triton
import triton.language as tl

# Fused BatchNorm + ReLU pointwise kernel
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
    # Channel index = (flat_idx // HW) % C
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


# Fused Conv implicit GEMM kernel for 3x3 conv with padding=1
# We implement this as an implicit GEMM
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H, KERNEL_W,
    # Strides for x (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # Strides for w (OUT_C, IN_C, KH, KW) -> flattened as (OUT_C, IN_C*KH*KW)
    s_woc, s_wk,
    # Strides for out (NHWC layout)
    s_on, s_oh, s_ow, s_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W
    # N = OUT_C
    # K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W
    K = IN_C * KERNEL_H * KERNEL_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        ic_idx = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        mask_x = valid_m & valid_k & valid_ih & valid_iw & valid_batch

        x_ptrs = (x_ptr +
                  batch_idx[:, None] * s_xn +
                  ic_idx[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (OUT_C, IN_C, KH, KW) stored as (OUT_C, K)
        # w_ptr[oc, k] = w_ptr + oc * s_woc + k * s_wk
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = w_ptr + offs_k[:, None] * s_wk + offs_n[None, :] * s_woc
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output (NHWC)
    valid_out_m = offs_m[:, None] < M
    valid_out_n = offs_n[None, :] < OUT_C
    mask_out = valid_out_m & valid_out_n

    out_ptrs = (out_ptr +
                batch_idx[:, None] * s_on +
                out_h_idx[:, None] * s_oh +
                out_w_idx[:, None] * s_ow +
                offs_n[None, :] * s_oc)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
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


def conv2d_triton(x_nhwc, w, OUT_H, OUT_W, STRIDE_H=1, STRIDE_W=1, PAD_H=1, PAD_W=1):
    """
    returns: (N, OUT_H, OUT_W, OUT_C) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape

    # Reshape w to (OUT_C, K) where K = IN_C * KH * KW
    # But we need to match the decode order: K -> (ic, kh, kw)
    # offs_k = ic * KH * KW + kh * KW + kw  (ic is outermost)
    # So w should be indexed as w[oc, ic, kh, kw] = w_flat[oc, ic*KH*KW + kh*KW + kw]
    w_flat = w.reshape(OUT_C, IN_C * KH * KW)  # (OUT_C, K)

    out = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=x_nhwc.dtype)

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    # Strides for x_nhwc (N, H, W, C)
    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    # Strides for w_flat (OUT_C, K)
    s_woc, s_wk = w_flat.stride()
    # Strides for out (N, OUT_H, OUT_W, OUT_C)
    s_on, s_oh, s_ow, s_oc = out.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_flat, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        KH, KW,
        s_xn, s_xh, s_xw, s_xc,
        s_woc, s_wk,
        s_on, s_oh, s_ow, s_oc,
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
    # x: [10, 32, 224, 224]
    N, C0, H, W = x.shape  # 10, 32, 224, 224

    # Layer 0
    bn0_out = bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
    # Convert to NHWC for conv
    bn0_nhwc = bn0_out.permute(0, 2, 3, 1).contiguous()
    conv0_nhwc = conv2d_triton(bn0_nhwc, p_layers_0_2_weight, H, W)
    conv0_out = conv0_nhwc.permute(0, 3, 1, 2).contiguous()
    # dropout(0.0) is identity
    # cat([x, conv0_out], dim=1) -> [10, 64, 224, 224]
    x1 = torch.cat([x, conv0_out], dim=1)

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3) + Cat -> [10, 96, 224, 224]
    bn1_out = bn_relu(x1, b_layers_1_0_running_mean, b_layers_1_0_running_var,
    bn1_nhwc = bn1_out.permute(0, 2, 3, 1).contiguous()
    conv1_nhwc = conv2d_triton(bn1_nhwc, p_layers_1_2_weight, H, W)
    conv1_out = conv1_nhwc.permute(0, 3, 1, 2).contiguous()
    x2 = torch.cat([x1, conv1_out], dim=1)

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3) + Cat -> [10, 128, 224, 224]
    bn2_out = bn_relu(x2, b_layers_2_0_running_mean, b_layers_2_0_running_var,
    bn2_nhwc = bn2_out.permute(0, 2, 3, 1).contiguous()
    conv2_nhwc = conv2d_triton(bn2_nhwc, p_layers_2_2_weight, H, W)
    conv2_out = conv2_nhwc.permute(0, 3, 1, 2).contiguous()
    x3 = torch.cat([x2, conv2_out], dim=1)

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3) + Cat -> [10, 160, 224, 224]
    bn3_out = bn_relu(x3, b_layers_3_0_running_mean, b_layers_3_0_running_var,
    bn3_nhwc = bn3_out.permute(0, 2, 3, 1).contiguous()
    conv3_nhwc = conv2d_triton(bn3_nhwc, p_layers_3_2_weight, H, W)
    conv3_out = conv3_nhwc.permute(0, 3, 1, 2).contiguous()
    x4 = torch.cat([x3, conv3_out], dim=1)

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3) + Cat -> [10, 192, 224, 224]
    bn4_out = bn_relu(x4, b_layers_4_0_running_mean, b_layers_4_0_running_var,
    bn4_nhwc = bn4_out.permute(0, 2, 3, 1).contiguous()
    conv4_nhwc = conv2d_triton(bn4_nhwc, p_layers_4_2_weight, H, W)
    conv4_out = conv4_nhwc.permute(0, 3, 1, 2).contiguous()
    x5 = torch.cat([x4, conv4_out], dim=1)

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3) + Cat -> [10, 224, 224, 224]
    bn5_out = bn_relu(x5, b_layers_5_0_running_mean, b_layers_5_0_running_var,
    bn5_nhwc = bn5_out.permute(0, 2, 3, 1).contiguous()
    conv5_nhwc = conv2d_triton(bn5_nhwc, p_layers_5_2_weight, H, W)
    conv5_out = conv5_nhwc.permute(0, 3, 1, 2).contiguous()
    x6 = torch.cat([x5, conv5_out], dim=1)

    return x6


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    num_layers = 6

    p_layers_0_0_weight = torch.rand(32, device='cuda')
    p_layers_0_0_bias = torch.rand(32, device='cuda')
    p_layers_0_2_weight = torch.rand(32, 32, 3, 3, device='cuda')

    p_layers_1_0_weight = torch.rand(64, device='cuda')
    p_layers_1_0_bias = torch.rand(64, device='cuda')
    p_layers_1_2_weight = torch.rand(32, 64, 3, 3, device='cuda')

    p_layers_2_0_weight = torch.rand(96, device='cuda')
    p_layers_2_0_bias = torch.rand(96, device='cuda')
    p_layers_2_2_weight = torch.rand(32, 96, 3, 3, device='cuda')

    p_layers_3_0_weight = torch.rand(128, device='cuda')
    p_layers_3_0_bias = torch.rand(128, device='cuda')
    p_layers_3_2_weight = torch.rand(32, 128, 3, 3, device='cuda')

    p_layers_4_0_weight = torch.rand(160, device='cuda')
    p_layers_4_0_bias = torch.rand(160, device='cuda')
    p_layers_4_2_weight = torch.rand(32, 160, 3, 3, device='cuda')

    p_layers_5_0_weight = torch.rand(192, device='cuda')
    p_layers_5_0_bias = torch.rand(192, device='cuda')
    p_layers_5_2_weight = torch.rand(32, 192, 3, 3, device='cuda')

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

        # Set weights from inputs
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

        weights = [
            (p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
             b_layers_0_0_running_mean, b_layers_0_0_running_var),
            (p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
             b_layers_1_0_running_mean, b_layers_1_0_running_var),
            (p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
             b_layers_2_0_running_mean, b_layers_2_0_running_var),
            (p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
             b_layers_3_0_running_mean, b_layers_3_0_running_var),
            (p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
             b_layers_4_0_running_mean, b_layers_4_0_running_var),
            (p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
             b_layers_5_0_running_mean, b_layers_5_0_running_var),
        ]

        for i, (w, b, cw, rm, rv) in enumerate(weights):
            model.layers[i][0].weight.data.copy_(w)
            model.layers[i][0].bias.data.copy_(b)
            model.layers[i][2].weight.data.copy_(cw)
            model.layers[i][0].running_mean.data.copy_(rm)
            model.layers[i][0].running_var.data.copy_(rv)

        with torch.no_grad():
            ref = model(x)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")