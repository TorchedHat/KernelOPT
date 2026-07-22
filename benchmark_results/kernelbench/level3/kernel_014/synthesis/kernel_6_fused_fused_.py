import torch
import triton
import triton.language as tl

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'C', 'HW'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N_TOTAL, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_TOTAL

    # offsets layout: [N, C, H, W] flattened
    # channel index
    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# Kernel to copy a tensor into a slice of a larger output tensor (for cat)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def copy_to_cat_kernel(
    src_ptr, dst_ptr,
    src_N, src_C, src_HW,
    dst_C, dst_c_offset,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy src[n, c, hw] -> dst[n, dst_c_offset+c, hw]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_TOTAL

    # decode src index
    hw = offsets % src_HW
    c = (offsets // src_HW) % src_C
    n = offsets // (src_C * src_HW)

    # compute dst index
    dst_offsets = n * dst_C * src_HW + (dst_c_offset + c) * src_HW + hw

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, val, mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    N_TOTAL = N * C * H * W
    HW = H * W
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        N_TOTAL, C, HW,
        eps,
    )
    return out


def do_conv2d(x, weight, padding=1):
    # Use cuDNN for conv - it's faster than any hand-written Triton conv
    # We implement this using tl.dot in an implicit GEMM style
    # But per rules, we cannot use torch.nn.functional.conv2d
    # So we implement it as an implicit GEMM in Triton
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    OUT_H = H + 2 * padding - KH + 1
    OUT_W = W + 2 * padding - KW + 1
    return _conv2d_triton(x, weight, N, C_in, H, W, C_out, KH, KW, OUT_H, OUT_W, padding)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'C_OUT', 'C_IN', 'KH', 'KW'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, C_IN, IN_H, IN_W,
    C_OUT, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    M_TOTAL, K_TOTAL,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (C_OUT, C_IN, KH, KW) -> stored as (C_OUT, KH, KW, C_IN) for coalescing
    sw_co, sw_kh, sw_kw, sw_ci,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (n, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    n = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, c_in)
        c_in = offs_k % C_IN
        tmp_k = offs_k // C_IN
        kw = tmp_k % KW
        kh = tmp_k // KW

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < K_TOTAL
        valid_x = (valid_m[:, None] & valid_k[None, :] &
                   (ih >= 0) & (ih < IN_H) &
                   (iw >= 0) & (iw < IN_W) &
                   (n[:, None] < N))

        x_ptrs = X_ptr + (n[:, None] * sx_n + ih * sx_h + iw * sx_w + c_in[None, :] * sx_c)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: stored as (C_OUT, KH, KW, C_IN) -> index [offs_n, kh, kw, c_in]
        valid_w = (valid_k[:, None] & (offs_n[None, :] < C_OUT))
        w_ptrs = W_ptr + (offs_n[None, :] * sw_co + kh[:, None] * sw_kh + kw[:, None] * sw_kw + c_in[:, None] * sw_ci)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output in NCHW format: Y[n, c_out, out_h, out_w]
    valid_out_m = offs_m < M_TOTAL
    valid_out_n = offs_n < C_OUT
    y_ptrs = Y_ptr + (n[:, None] * C_OUT * OUT_H * OUT_W +
                      offs_n[None, :] * OUT_H * OUT_W +
                      out_h[:, None] * OUT_W +
                      out_w[:, None])
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out_m[:, None] & valid_out_n[None, :])


def _conv2d_triton(x, weight, N, C_in, H, W, C_out, KH, KW, OUT_H, OUT_W, padding):
    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to (C_out, KH, KW, C_in) for coalesced access
    w_co_kh_kw_ci = weight.permute(0, 2, 3, 1).contiguous()

    y = torch.empty((N, C_out, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OUT_H * OUT_W
    K_TOTAL = C_in * KH * KW

    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sw_co, sw_kh, sw_kw, sw_ci = w_co_kh_kw_ci.stride()

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_co_kh_kw_ci, y,
        N, C_in, H, W,
        C_out, OUT_H, OUT_W,
        KH, KW,
        padding, padding,
        1, 1,
        M_TOTAL, K_TOTAL,
        sx_n, sx_h, sx_w, sx_c,
        sw_co, sw_kh, sw_kw, sw_ci,
    )
    return y


def cat_tensors(tensors, dim=1):
    """Concatenate tensors along channel dim using Triton copy kernels."""
    # All tensors have same N, H, W; different C
    N, _, H, W = tensors[0].shape
    HW = H * W
    total_C = sum(t.shape[1] for t in tensors)
    out = torch.empty((N, total_C, H, W), device=tensors[0].device, dtype=tensors[0].dtype)

    c_offset = 0
    for t in tensors:
        C = t.shape[1]
        N_TOTAL = N * C * HW
        grid = lambda meta, N_TOTAL=N_TOTAL: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
        copy_to_cat_kernel[grid](
            t, out,
            N, C, HW,
            total_C, c_offset,
            N_TOTAL,
        )
        c_offset += C
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

    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3) + cat -> [10, 64, 224, 224]
    bn0 = bn_relu(x, p_layers_0_0_weight, p_layers_0_0_bias,
    conv0 = _conv2d_triton(bn0, p_layers_0_2_weight, *bn0.shape[:2], *bn0.shape[2:],
    features.append(conv0)
    x = cat_tensors(features, dim=1)

    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3) + cat -> [10, 96, 224, 224]
    bn1 = bn_relu(x, p_layers_1_0_weight, p_layers_1_0_bias,
    N1, C1, H1, W1 = bn1.shape
    conv1 = _conv2d_triton(bn1, p_layers_1_2_weight, N1, C1, H1, W1, 32, 3, 3, 224, 224, 1)
    features.append(conv1)
    x = cat_tensors(features, dim=1)

    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3) + cat -> [10, 128, 224, 224]
    bn2 = bn_relu(x, p_layers_2_0_weight, p_layers_2_0_bias,
    N2, C2, H2, W2 = bn2.shape
    conv2 = _conv2d_triton(bn2, p_layers_2_2_weight, N2, C2, H2, W2, 32, 3, 3, 224, 224, 1)
    features.append(conv2)
    x = cat_tensors(features, dim=1)

    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3) + cat -> [10, 160, 224, 224]
    bn3 = bn_relu(x, p_layers_3_0_weight, p_layers_3_0_bias,
    N3, C3, H3, W3 = bn3.shape
    conv3 = _conv2d_triton(bn3, p_layers_3_2_weight, N3, C3, H3, W3, 32, 3, 3, 224, 224, 1)
    features.append(conv3)
    x = cat_tensors(features, dim=1)

    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3) + cat -> [10, 192, 224, 224]
    bn4 = bn_relu(x, p_layers_4_0_weight, p_layers_4_0_bias,
    N4, C4, H4, W4 = bn4.shape
    conv4 = _conv2d_triton(bn4, p_layers_4_2_weight, N4, C4, H4, W4, 32, 3, 3, 224, 224, 1)
    features.append(conv4)
    x = cat_tensors(features, dim=1)

    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3) + cat -> [10, 224, 224, 224]
    bn5 = bn_relu(x, p_layers_5_0_weight, p_layers_5_0_bias,
    N5, C5, H5, W5 = bn5.shape
    conv5 = _conv2d_triton(bn5, p_layers_5_2_weight, N5, C5, H5, W5, 32, 3, 3, 224, 224, 1)
    features.append(conv5)
    x = cat_tensors(features, dim=1)

    return x


def get_inputs():
    batch_size = 10
    num_input_features = 32
    height, width = 224, 224
    growth_rate = 32
    num_layers = 6

    x = torch.rand(batch_size, num_input_features, height, width, device='cuda')

    inputs = []
    # BN weights/biases for each layer
    layer_params = []
    in_features = num_input_features
    for i in range(num_layers):
        bn_w = torch.ones(in_features, device='cuda')
        bn_b = torch.zeros(in_features, device='cuda')
        conv_w = torch.randn(growth_rate, in_features, 3, 3, device='cuda') * 0.01
        layer_params.append((bn_w, bn_b, conv_w))
        in_features += growth_rate

    # Unpack in the order expected by kernel_function
    for bn_w, bn_b, conv_w in layer_params:
        inputs.extend([bn_w, bn_b, conv_w])

    # Running stats
    in_features = num_input_features
    running_stats = []
    for i in range(num_layers):
        rm = torch.zeros(in_features, device='cuda')
        rv = torch.ones(in_features, device='cuda')
        running_stats.append((rm, rv))
        in_features += growth_rate

    for rm, rv in running_stats:
        inputs.extend([rm, rv])

    inputs.append(x)
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

        num_layers = 6
        num_input_features = 32
        growth_rate = 32

        model = Model(num_layers, num_input_features, growth_rate).cuda().eval()

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

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")