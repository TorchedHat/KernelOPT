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
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, num_channels, spatial,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # channel index: layout is (N, C, H, W) so channel = (offset / spatial) % num_channels
    chan = (offsets // spatial) % num_channels
    chan_mask = chan < num_channels

    mean = tl.load(mean_ptr + chan, mask=mask & chan_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + chan, mask=mask & chan_mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + chan, mask=mask & chan_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + chan, mask=mask & chan_mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if do_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5, relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, spatial,
        eps,
        do_relu=relu,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_relu_kernel[grid](x, out, numel)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_sigmoid_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_sigmoid_kernel[grid](x, out, numel)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_mul_kernel(
    x_ptr, scale_ptr, out_ptr,
    numel, spatial,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # scale has shape (N, C, 1, 1), broadcast over spatial
    scale_idx = offsets // spatial
    scale_mask = scale_idx < (numel // spatial)
    scale = tl.load(scale_ptr + scale_idx, mask=mask & scale_mask, other=1.0).to(tl.float32)
    out = x * scale
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple row-parallel linear for small M
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # We'll do this differently - use a flat approach
    # Actually for linear we need a proper matmul
    # This kernel handles one output element per thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask_col = cols < N

    # For each output (row, col), compute dot product over K
    # This is too slow for large K - use a different approach
    out_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(K):
        x_val = tl.load(x_ptr + row * K + k)
        w_vals = tl.load(w_ptr + cols * K + k, mask=mask_col, other=0.0)
        out_val += x_val * w_vals
    b_vals = tl.load(b_ptr + cols, mask=mask_col, other=0.0)
    out_val += b_vals
    tl.store(out_ptr + row * N + cols, out_val, mask=mask_col)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b_vals[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=out_mask)


def apply_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)


def adaptive_avg_pool2d(x, output_size):
    import torch.nn.functional as F
    return F.adaptive_avg_pool2d(x, output_size)


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight,
    p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight, p_mbconv2_9_weight,
    p_mbconv2_11_weight, p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight, p_mbconv3_9_weight,
    p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight, p_mbconv4_9_weight,
    p_mbconv4_11_weight, p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight, p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight, p_mbconv5_9_weight,
    p_mbconv5_11_weight, p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight, p_bn_final_weight, p_bn_final_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
    b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
    b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
    b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
    b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
    b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
    b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
    b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
    b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
    b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
    b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
    b_mbconv4_12_running_mean, b_mbconv4_12_running_var,
    b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
    b_mbconv5_4_running_mean, b_mbconv5_4_running_var,
    b_mbconv5_12_running_mean, b_mbconv5_12_running_var,
    b_bn_final_running_mean, b_bn_final_running_var,
    x,
):
    # Initial conv + bn + relu
    # conv2d: (2,3,224,224) -> (2,32,112,112)
    out = conv2d(x, p_conv1_weight, stride=2, padding=1)
    out = apply_bn_relu(out, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var, relu=True)

    # MBConv1: in=32, out=96, stride=1, expand=3 (32*3=96)
    # Expansion conv
    t = conv2d(out, p_mbconv1_0_weight)
    t = apply_bn_relu(t, p_mbconv1_1_weight, p_mbconv1_1_bias, b_mbconv1_1_running_mean, b_mbconv1_1_running_var, relu=True)
    # Depthwise conv
    t = conv2d(t, p_mbconv1_3_weight, stride=1, padding=1, groups=96)
    t = apply_bn_relu(t, p_mbconv1_4_weight, p_mbconv1_4_bias, b_mbconv1_4_running_mean, b_mbconv1_4_running_var, relu=True)
    # SE
    se = adaptive_avg_pool2d(t, (1, 1))
    se = conv2d(se, p_mbconv1_7_weight)
    se = apply_relu(se)
    se = conv2d(se, p_mbconv1_9_weight)
    se = apply_sigmoid(se)
    # Scale: t * se (broadcast)
    N, C, H, W = t.shape
    se_flat = se.view(N * C)
    t_flat = t.reshape(N * C * H * W)
    out_flat = torch.empty_like(t_flat)
    numel = N * C * H * W
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_mul_kernel[grid](t_flat, se_flat, out_flat, numel, spatial, C)
    t = out_flat.reshape(N, C, H, W)
    # Output conv + bn (no relu)
    t = conv2d(t, p_mbconv1_11_weight)
    t = apply_bn_relu(t, p_mbconv1_12_weight, p_mbconv1_12_bias, b_mbconv1_12_running_mean, b_mbconv1_12_running_var, relu=False)

    # MBConv2: in=96, out=144, stride=2, expand=6 (96*6=576)
    t2 = conv2d(t, p_mbconv2_0_weight)
    t2 = apply_bn_relu(t2, p_mbconv2_1_weight, p_mbconv2_1_bias, b_mbconv2_1_running_mean, b_mbconv2_1_running_var, relu=True)
    t2 = conv2d(t2, p_mbconv2_3_weight, stride=2, padding=1, groups=576)
    t2 = apply_bn_relu(t2, p_mbconv2_4_weight, p_mbconv2_4_bias, b_mbconv2_4_running_mean, b_mbconv2_4_running_var, relu=True)
    # SE
    se2 = adaptive_avg_pool2d(t2, (1, 1))
    se2 = conv2d(se2, p_mbconv2_7_weight)
    se2 = apply_relu(se2)
    se2 = conv2d(se2, p_mbconv2_9_weight)
    se2 = apply_sigmoid(se2)
    N2, C2, H2, W2 = t2.shape
    se2_flat = se2.view(N2 * C2)
    t2_flat = t2.reshape(N2 * C2 * H2 * W2)
    out2_flat = torch.empty_like(t2_flat)
    numel2 = N2 * C2 * H2 * W2
    spatial2 = H2 * W2
    fused_mul_kernel[lambda meta: (triton.cdiv(numel2, meta['BLOCK_SIZE']),)](
    t2 = out2_flat.reshape(N2, C2, H2, W2)
    t2 = conv2d(t2, p_mbconv2_11_weight)
    t2 = apply_bn_relu(t2, p_mbconv2_12_weight, p_mbconv2_12_bias, b_mbconv2_12_running_mean, b_mbconv2_12_running_var, relu=False)

    # MBConv3: in=144, out=192, stride=2, expand=6 (144*6=864)
    t3 = conv2d(t2, p_mbconv3_0_weight)
    t3 = apply_bn_relu(t3, p_mbconv3_1_weight, p_mbconv3_1_bias, b_mbconv3_1_running_mean, b_mbconv3_1_running_var, relu=True)
    t3 = conv2d(t3, p_mbconv3_3_weight, stride=2, padding=1, groups=864)
    t3 = apply_bn_relu(t3, p_mbconv3_4_weight, p_mbconv3_4_bias, b_mbconv3_4_running_mean, b_mbconv3_4_running_var, relu=True)
    se3 = adaptive_avg_pool2d(t3, (1, 1))
    se3 = conv2d(se3, p_mbconv3_7_weight)
    se3 = apply_relu(se3)
    se3 = conv2d(se3, p_mbconv3_9_weight)
    se3 = apply_sigmoid(se3)
    N3, C3, H3, W3 = t3.shape
    se3_flat = se3.view(N3 * C3)
    t3_flat = t3.reshape(N3 * C3 * H3 * W3)
    out3_flat = torch.empty_like(t3_flat)
    numel3 = N3 * C3 * H3 * W3
    spatial3 = H3 * W3
    fused_mul_kernel[lambda meta: (triton.cdiv(numel3, meta['BLOCK_SIZE']),)](
    t3 = out3_flat.reshape(N3, C3, H3, W3)
    t3 = conv2d(t3, p_mbconv3_11_weight)
    t3 = apply_bn_relu(t3, p_mbconv3_12_weight, p_mbconv3_12_bias, b_mbconv3_12_running_mean, b_mbconv3_12_running_var, relu=False)

    # MBConv4: in=192, out=288, stride=2, expand=6 (192*6=1152)
    t4 = conv2d(t3, p_mbconv4_0_weight)
    t4 = apply_bn_relu(t4, p_mbconv4_1_weight, p_mbconv4_1_bias, b_mbconv4_1_running_mean, b_mbconv4_1_running_var, relu=True)
    t4 = conv2d(t4, p_mbconv4_3_weight, stride=2, padding=1, groups=1152)
    t4 = apply_bn_relu(t4, p_mbconv4_4_weight, p_mbconv4_4_bias, b_mbconv4_4_running_mean, b_mbconv4_4_running_var, relu=True)
    se4 = adaptive_avg_pool2d(t4, (1, 1))
    se4 = conv2d(se4, p_mbconv4_7_weight)
    se4 = apply_relu(se4)
    se4 = conv2d(se4, p_mbconv4_9_weight)
    se4 = apply_sigmoid(se4)
    N4, C4, H4, W4 = t4.shape
    se4_flat = se4.view(N4 * C4)
    t4_flat = t4.reshape(N4 * C4 * H4 * W4)
    out4_flat = torch.empty_like(t4_flat)
    numel4 = N4 * C4 * H4 * W4
    spatial4 = H4 * W4
    fused_mul_kernel[lambda meta: (triton.cdiv(numel4, meta['BLOCK_SIZE']),)](
    t4 = out4_flat.reshape(N4, C4, H4, W4)
    t4 = conv2d(t4, p_mbconv4_11_weight)
    t4 = apply_bn_relu(t4, p_mbconv4_12_weight, p_mbconv4_12_bias, b_mbconv4_12_running_mean, b_mbconv4_12_running_var, relu=False)

    # MBConv5: in=288, out=384, stride=1, expand=6 (288*6=1728)
    t5 = conv2d(t4, p_mbconv5_0_weight)
    t5 = apply_bn_relu(t5, p_mbconv5_1_weight, p_mbconv5_1_bias, b_mbconv5_1_running_mean, b_mbconv5_1_running_var, relu=True)
    t5 = conv2d(t5, p_mbconv5_3_weight, stride=1, padding=1, groups=1728)
    t5 = apply_bn_relu(t5, p_mbconv5_4_weight, p_mbconv5_4_bias, b_mbconv5_4_running_mean, b_mbconv5_4_running_var, relu=True)
    se5 = adaptive_avg_pool2d(t5, (1, 1))
    se5 = conv2d(se5, p_mbconv5_7_weight)
    se5 = apply_relu(se5)
    se5 = conv2d(se5, p_mbconv5_9_weight)
    se5 = apply_sigmoid(se5)
    N5, C5, H5, W5 = t5.shape
    se5_flat = se5.view(N5 * C5)
    t5_flat = t5.reshape(N5 * C5 * H5 * W5)
    out5_flat = torch.empty_like(t5_flat)
    numel5 = N5 * C5 * H5 * W5
    spatial5 = H5 * W5
    fused_mul_kernel[lambda meta: (triton.cdiv(numel5, meta['BLOCK_SIZE']),)](
    t5 = out5_flat.reshape(N5, C5, H5, W5)
    t5 = conv2d(t5, p_mbconv5_11_weight)
    t5 = apply_bn_relu(t5, p_mbconv5_12_weight, p_mbconv5_12_bias, b_mbconv5_12_running_mean, b_mbconv5_12_running_var, relu=False)

    # Final conv + bn + relu
    final = conv2d(t5, p_conv_final_weight)
    final = apply_bn_relu(final, p_bn_final_weight, p_bn_final_bias, b_bn_final_running_mean, b_bn_final_running_var, relu=True)

    # Global average pool
    final = adaptive_avg_pool2d(final, (1, 1))

    # Flatten
    final = final.reshape(final.shape[0], -1)

    # Linear
    out_fc = apply_linear(final, p_fc_weight, p_fc_bias)

    return out_fc


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'

    def rand(*shape):
        return torch.randn(*shape, device=device)

    def ones(shape):
        return torch.ones(shape, device=device)

    def zeros(shape):
        return torch.zeros(shape, device=device)

    return [
        rand(32, 3, 3, 3),      # p_conv1_weight
        ones(32),                # p_bn1_weight
        zeros(32),               # p_bn1_bias
        rand(96, 32, 1, 1),     # p_mbconv1_0_weight
        ones(96),                # p_mbconv1_1_weight
        zeros(96),               # p_mbconv1_1_bias
        rand(96, 1, 3, 3),      # p_mbconv1_3_weight
        ones(96),                # p_mbconv1_4_weight
        zeros(96),               # p_mbconv1_4_bias
        rand(24, 96, 1, 1),     # p_mbconv1_7_weight
        rand(96, 24, 1