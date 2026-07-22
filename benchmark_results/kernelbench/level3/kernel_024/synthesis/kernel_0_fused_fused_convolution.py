import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm + ReLU kernel (pointwise, applied after conv)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    eps,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # channel index
    c = (offsets // HW) % C

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b

    if apply_relu:
        y = tl.maximum(y, 0.0)

    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5, relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
        numel, C, HW, eps,
        apply_relu=relu,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fused ReLU kernel (pointwise)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fused Sigmoid kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


def fused_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fused AdaptiveAvgPool2d (global, output 1x1) kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, c) pair
    nc = tl.program_id(0)
    n = nc // C
    c = nc % C

    base = n * C * HW + c * HW
    acc = tl.zeros((), dtype=tl.float32)

    for off in range(0, HW, BLOCK_HW):
        offsets = off + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)

    out = acc / HW
    tl.store(out_ptr + nc, out)


def fused_adaptive_avg_pool(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool_kernel[grid](x, out, N, C, HW)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fused Linear (matmul + bias) kernel using tl.dot
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=mask_out)


def fused_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Conv2d helpers (using PyTorch's cuDNN -- fastest for standard convolutions)
# ─────────────────────────────────────────────────────────────────────────────

def conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
    return torch.nn.functional.conv2d(x, weight, bias=bias, stride=stride,
                                       padding=padding, groups=groups)


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function implementing EfficientNetB2 forward pass
# ─────────────────────────────────────────────────────────────────────────────

def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight,
    p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight,
    p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight,
    p_mbconv1_9_weight,
    p_mbconv1_11_weight,
    p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight,
    p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight,
    p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight,
    p_mbconv2_9_weight,
    p_mbconv2_11_weight,
    p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight,
    p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight,
    p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight,
    p_mbconv3_9_weight,
    p_mbconv3_11_weight,
    p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight,
    p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight,
    p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight,
    p_mbconv4_9_weight,
    p_mbconv4_11_weight,
    p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight,
    p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight,
    p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight,
    p_mbconv5_9_weight,
    p_mbconv5_11_weight,
    p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight,
    p_bn_final_weight, p_bn_final_bias,
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
    # ── Stem ──────────────────────────────────────────────────────────────────
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = fused_bn_relu(x, b_bn1_running_mean, b_bn1_running_var,
                      p_bn1_weight, p_bn1_bias, relu=True)

    # ── MBConv1 (expand=1, stride=1, in=32, out=96) ───────────────────────────
    # expand phase (expand_ratio=1 means no expansion conv, but looking at weights
    # p_mbconv1_0_weight is [96,32,1,1] so expand_ratio=3 effectively)
    # Actually from the weight shapes: mbconv1_0 is [96,32,1,1] -> expand conv
    # mbconv1_3 is [96,1,3,3] -> depthwise
    # mbconv1_7 is [24,96,1,1] -> SE squeeze
    # mbconv1_9 is [96,24,1,1] -> SE excite
    # mbconv1_11 is [96,96,1,1] -> output conv (but wait, out_channels=96 not 24?)
    # Actually looking at the computation graph more carefully:
    # The first mbconv block: in=32, out=96 (but graph shows output of mbconv1_11 -> [2,96,1,1])
    # Wait, looking at the graph: after SE, conv2d -> [2,96,1,1] then bn -> [2,96,1,1]
    # Then conv2d -> [2,576,1,1]... so mbconv1 output is 96 channels
    # But p_mbconv1_11_weight is [96,96,1,1] which maps expanded->out_channels
    # Hmm, but the model says out_channels=96 for mbconv1... let me re-read
    # _make_mbconv_block(32, 96, 1, 3): in=32, out=96, stride=1, expand=3
    # expanded = 32*3 = 96
    # So: expand conv [96,32,1,1], dw [96,1,3,3], SE squeeze [24,96,1,1], SE excite [96,24,1,1]
    # output conv [96,96,1,1] -> but wait that maps expanded(96)->out(96)
    # Actually the output conv should be [out_channels, expanded_channels, 1, 1] = [96, 96, 1, 1] ✓

    # Expand
    t = conv2d(x, p_mbconv1_0_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
                      p_mbconv1_1_weight, p_mbconv1_1_bias, relu=True)
    # Depthwise
    t = conv2d(t, p_mbconv1_3_weight, stride=1, padding=1, groups=t.shape[1])
    t = fused_bn_relu(t, b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
                      p_mbconv1_4_weight, p_mbconv1_4_bias, relu=True)
    # SE
    se = fused_adaptive_avg_pool(t)
    se = conv2d(se, p_mbconv1_7_weight, stride=1, padding=0)
    se = fused_relu(se)
    se = conv2d(se, p_mbconv1_9_weight, stride=1, padding=0)
    se = fused_sigmoid(se)
    t = t * se
    # Output
    t = conv2d(t, p_mbconv1_11_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
                      p_mbconv1_12_weight, p_mbconv1_12_bias, relu=False)
    x = t  # no residual for mbconv1 (stride=1 but in!=out... actually 32!=96 so no skip)

    # ── MBConv2 (expand=6, stride=2, in=96, out=144) ──────────────────────────
    # p_mbconv2_0_weight: [576,96,1,1] expand
    # p_mbconv2_3_weight: [576,1,3,3] depthwise stride=2
    # p_mbconv2_7_weight: [144,576,1,1] SE squeeze
    # p_mbconv2_9_weight: [576,144,1,1] SE excite
    # p_mbconv2_11_weight: [144,576,1,1] output
    t = conv2d(x, p_mbconv2_0_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
                      p_mbconv2_1_weight, p_mbconv2_1_bias, relu=True)
    t = conv2d(t, p_mbconv2_3_weight, stride=2, padding=1, groups=t.shape[1])
    t = fused_bn_relu(t, b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
                      p_mbconv2_4_weight, p_mbconv2_4_bias, relu=True)
    se = fused_adaptive_avg_pool(t)
    se = conv2d(se, p_mbconv2_7_weight, stride=1, padding=0)
    se = fused_relu(se)
    se = conv2d(se, p_mbconv2_9_weight, stride=1, padding=0)
    se = fused_sigmoid(se)
    t = t * se
    t = conv2d(t, p_mbconv2_11_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
                      p_mbconv2_12_weight, p_mbconv2_12_bias, relu=False)
    x = t  # no residual (stride=2)

    # ── MBConv3 (expand=6, stride=2, in=144, out=192) ─────────────────────────
    t = conv2d(x, p_mbconv3_0_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
                      p_mbconv3_1_weight, p_mbconv3_1_bias, relu=True)
    t = conv2d(t, p_mbconv3_3_weight, stride=2, padding=1, groups=t.shape[1])
    t = fused_bn_relu(t, b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
                      p_mbconv3_4_weight, p_mbconv3_4_bias, relu=True)
    se = fused_adaptive_avg_pool(t)
    se = conv2d(se, p_mbconv3_7_weight, stride=1, padding=0)
    se = fused_relu(se)
    se = conv2d(se, p_mbconv3_9_weight, stride=1, padding=0)
    se = fused_sigmoid(se)
    t = t * se
    t = conv2d(t, p_mbconv3_11_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
                      p_mbconv3_12_weight, p_mbconv3_12_bias, relu=False)
    x = t

    # ── MBConv4 (expand=6, stride=2, in=192, out=288) ─────────────────────────
    t = conv2d(x, p_mbconv4_0_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
                      p_mbconv4_1_weight, p_mbconv4_1_bias, relu=True)
    t = conv2d(t, p_mbconv4_3_weight, stride=2, padding=1, groups=t.shape[1])
    t = fused_bn_relu(t, b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
                      p_mbconv4_4_weight, p_mbconv4_4_bias, relu=True)
    se = fused_adaptive_avg_pool(t)
    se = conv2d(se, p_mbconv4_7_weight, stride=1, padding=0)
    se = fused_relu(se)
    se = conv2d(se, p_mbconv4_9_weight, stride=1, padding=0)
    se = fused_sigmoid(se)
    t = t * se
    t = conv2d(t, p_mbconv4_11_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv4_12_running_mean, b_mbconv4_12_running_var,
                      p_mbconv4_12_weight, p_mbconv4_12_bias, relu=False)
    x = t

    # ── MBConv5 (expand=6, stride=1, in=288, out=384) ─────────────────────────
    t = conv2d(x, p_mbconv5_0_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
                      p_mbconv5_1_weight, p_mbconv5_1_bias, relu=True)
    t = conv2d(t, p_mbconv5_3_weight, stride=1, padding=1, groups=t.shape[1])
    t = fused_bn_relu(t, b_mbconv5_4_running_mean, b_mbconv5_4_running_var,
                      p_mbconv5_4_weight, p_mbconv5_4_bias, relu=True)
    se = fused_adaptive_avg_pool(t)
    se = conv2d(se, p_mbconv5_7_weight, stride=1, padding=0)
    se = fused_relu(se)
    se = conv2d(se, p_mbconv5_9_weight, stride=1, padding=0)
    se = fused_sigmoid(se)
    t = t * se
    t = conv2d(t, p_mbconv5_11_weight, stride=1, padding=0)
    t = fused_bn_relu(t, b_mbconv5_12_running_mean, b_mbconv5_12_running_var,
                      p_mbconv5_12_weight, p_mbconv5_12_bias, relu=False)
    x = t

    # ── Final conv + BN + ReLU ────────────────────────────────────────────────
    x = conv2d(x, p_conv_final_weight, stride=1, padding=0)
    x = fused_bn_relu(x, b_bn_final_running_mean, b_bn_final_running_var,
                      p_bn_final_weight, p_bn_final_bias, relu=True)

    # ── Global average pool + flatten ─────────────────────────────────────────
    x = fused_adaptive_avg_pool(x)
    x = x.view(x.shape[0], -1)  # flatten

    # ── FC layer ──────────────────────────────────────────────────────────────
    x = fused_linear(x, p_fc_weight, p_fc_bias)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Sample inputs
# ─────────────────────────────────────────────────────────────────────────────

def get_inputs():
    device = 'cuda'
    dtype = torch.float32

    def t(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    def ones(shape):
        return torch.ones(shape, device=device, dtype=dtype)

    def zeros(shape):
        return torch.zeros(shape, device=device, dtype=dtype)

    return [
        t(32, 3, 3, 3),      # p_conv1_weight
        ones(32), zeros(32), # p_bn1_weight, p_bn1_bias
        t(96, 32, 1, 1),     # p_mbconv1_0_weight
        ones(96), zeros(96), # p_mbconv1_1_weight, p_mbconv1_1_bias
        t(96, 1, 3, 3),      # p_mbconv1_3_weight
        ones(96), zeros(96), # p_mbconv1_4_weight, p_mbconv1_4_bias
        t(24, 96, 1, 1),     # p_mbconv1_7_weight
        t(96, 24, 1, 1),     # p_mbconv1_9_weight
        t(96, 96, 1, 1),     # p_mbconv1_11_weight
        ones(96), zeros(96), # p_mbconv1_12_weight