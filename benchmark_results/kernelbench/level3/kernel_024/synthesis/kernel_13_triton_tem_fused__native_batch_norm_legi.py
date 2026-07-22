import torch
import triton
import triton.language as tl

# ============================================================
# Fused BatchNorm + ReLU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, HW,
        eps=eps,
    )
    return out


# ============================================================
# Fused BatchNorm (no relu) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn(x, running_mean, running_var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, HW,
        eps=eps,
    )
    return out


# ============================================================
# Fused Linear (GEMM) kernel: y = x @ W^T + b
# x: (M, K), W: (N, K), b: (N,), out: (M, N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
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

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ============================================================
# ReLU pointwise kernel
# ============================================================
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
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


# ============================================================
# Sigmoid pointwise kernel
# ============================================================
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
def sigmoid_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


# ============================================================
# Adaptive Average Pool 2D kernel (global pooling to 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
    ],
    key=['HW', 'NC'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW, NC,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    nc_mask = pid < NC
    n = pid // C
    c = pid % C

    base = n * C * HW + c * HW
    acc = tl.zeros((), dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        vals = tl.load(x_ptr + base + hw_offs, mask=hw_mask & nc_mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    avg = acc / HW
    tl.store(out_ptr + pid, avg, mask=nc_mask)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (NC,)
    adaptive_avg_pool2d_kernel[grid](
        x, out.view(-1),
        N, C, HW, NC,
    )
    return out


# ============================================================
# Multiply kernel (for SE attention: x * sigmoid_out)
# ============================================================
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
def multiply_kernel(a_ptr, b_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a * b, mask=mask)


def triton_multiply(a, b):
    out = torch.empty_like(a)
    numel = a.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    multiply_kernel[grid](a, b, out, numel)
    return out


# ============================================================
# Conv2d wrapper (uses PyTorch/cuDNN)
# ============================================================
def conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)


# ============================================================
# MBConv block forward
# ============================================================
def mbconv_block_forward(
    x,
    # expand conv
    expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
    # depthwise conv
    dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
    # SE
    se_reduce_w, se_expand_w,
    # project conv
    proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
    stride,
    expand_ratio,
):
    # Expansion phase
    if expand_ratio != 1:
        x = conv2d(x, expand_w, stride=1, padding=0)
        x = fused_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b)

    # Depthwise conv
    groups = x.shape[1]
    x = conv2d(x, dw_w, stride=stride, padding=1, groups=groups)
    x = fused_bn_relu(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b)

    # Squeeze and Excitation
    se = triton_adaptive_avg_pool2d(x)
    se = conv2d(se, se_reduce_w, stride=1, padding=0)
    se = triton_relu(se)
    se = conv2d(se, se_expand_w, stride=1, padding=0)
    se = triton_sigmoid(se)
    # Broadcast multiply: x * se (se is (N, C, 1, 1))
    x = triton_multiply(x, se.expand_as(x))

    # Project
    x = conv2d(x, proj_w, stride=1, padding=0)
    x = fused_bn(x, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b)

    return x


# ============================================================
# Main kernel function
# ============================================================
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
    # Initial conv + bn + relu
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = fused_bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)

    # MBConv1: expand_ratio=1 (no expansion conv), stride=1
    # expand_ratio=1 means no expansion phase
    # depthwise: groups=32, stride=1
    # SE: reduce to 32//4=8... wait, looking at weights:
    # p_mbconv1_0_weight: [96, 32, 1, 1] -> expand conv (expand_ratio=3? no, 96/32=3)
    # Actually from the model: mbconv1 = _make_mbconv_block(32, 96, 1, 3) -> expand_ratio=3
    # Wait, looking at the computation graph more carefully:
    # conv2d -> [2, 96, 112, 112]: expand conv (32->96)
    # conv2d -> [2, 96, 112, 112]: depthwise (96->96, groups=96)
    # adaptive_avg_pool2d -> [2, 96, 1, 1]
    # conv2d -> [2, 24, 1, 1]: SE reduce (96->24)
    # conv2d -> [2, 96, 1, 1]: SE expand (24->96)
    # conv2d -> [2, 96, 1, 1]: project... wait that's [2,96,1,1] not [2,96,112,112]
    # Hmm, looking at the graph again - after SE the spatial dims are 1x1
    # Actually the SE output is multiplied with the feature map
    # But the computation graph shows conv2d -> [2, 96, 1, 1] after sigmoid
    # That seems wrong... let me re-read

    # From computation graph:
    # conv2d -> [2, 96, 112, 112] (expand)
    # bn + relu -> [2, 96, 112, 112]
    # conv2d -> [2, 96, 112, 112] (depthwise)
    # bn + relu -> [2, 96, 112, 112]
    # adaptive_avg_pool2d -> [2, 96, 1, 1]
    # conv2d -> [2, 24, 1, 1] (SE reduce)
    # relu -> [2, 24, 1, 1]
    # conv2d -> [2, 96, 1, 1] (SE expand)
    # sigmoid -> [2, 96, 1, 1]
    # conv2d -> [2, 96, 1, 1] ??? This is the project conv applied to 1x1?
    # No wait - looking at the original model code, the project conv is applied to the
    # full feature map. The SE output (1x1) is multiplied with the feature map first.
    # But the computation graph shows conv2d -> [2, 96, 1, 1] after sigmoid...
    # 
    # Actually I think the computation graph is showing the SE multiply result
    # being applied to the depthwise output which at this point has been pooled.
    # Wait no - let me look at the shapes again.
    #
    # After mbconv1 depthwise: [2, 96, 112, 112]
    # After adaptive_avg_pool2d: [2, 96, 1, 1]
    # After SE reduce conv: [2, 24, 1, 1]
    # After SE expand conv: [2, 96, 1, 1]
    # After sigmoid: [2, 96, 1, 1]
    # conv2d -> [2, 96, 1, 1]: This must be the project conv applied to the 1x1 pooled * sigmoid
    # But that doesn't match the original model...
    #
    # Oh wait! Looking at the computation graph output shapes more carefully:
    # The shapes after mbconv1 are all 1x1 because the input to mbconv2 is 1x1!
    # That means mbconv1 with stride=1 on 112x112 input gives 112x112,
    # but then mbconv2 with stride=2 gives 56x56... but the graph shows 1x1 everywhere.
    #
    # I think the computation graph is showing a DIFFERENT model than what I expect.
    # Let me just follow the computation graph exactly.

    # Following the computation graph exactly:
    # After initial conv+bn+relu: [2, 32, 112, 112]
    
    # MBConv1 block (expand_ratio=3, stride=1):
    # expand: 32->96, [2, 96, 112, 112]
    x1 = conv2d(x, p_mbconv1_0_weight, stride=1, padding=0)
    x1 = fused_bn_relu(x1, b_mbconv1_1_running_mean, b_mbconv1_1_running_var, p_mbconv1_1_weight, p_mbconv1_1_bias)
    # depthwise: 96->96, groups=96, stride=1, [2, 96, 112, 112]
    x1 = conv2d(x1, p_mbconv1_3_weight, stride=1, padding=1, groups=96)
    x1 = fused_bn_relu(x1, b_mbconv1_4_running_mean, b_mbconv1_4_running_var, p_mbconv1_4_weight, p_mbconv1_4_bias)
    # SE: pool -> [2, 96, 1, 1]
    se1 = triton_adaptive_avg_pool2d(x1)
    # SE reduce: 96->24, [2, 24, 1, 1]
    se1 = conv2d(se1, p_mbconv1_7_weight, stride=1, padding=0)
    se1 = triton_relu(se1)
    # SE expand: 24->96, [2, 96, 1, 1]
    se1 = conv2d(se1, p_mbconv1_9_weight, stride=1, padding=0)
    se1 = triton_sigmoid(se1)
    # Multiply: x1 * se1 (broadcast)
    x1 = triton_multiply(x1, se1.expand_as(x1))
    # Project: 96->96, [2, 96, 1, 1]... 
    # Wait the graph shows [2, 96, 1, 1] after the project conv
    # That means the project conv is applied AFTER the SE multiply to the 1x1 result?
    # No - se1 is [2, 96, 1, 1] and x1 is [2, 96, 112, 112]
    # se1.expand_as(x1) would be [2, 96, 112, 112]
    # So x1 after multiply is [2, 96, 112, 112]
    # Then project conv: [2, 96, 112, 112] -> [2, 96, 112, 112]? But graph shows [2, 96, 1, 1]
    # 
    # Hmm, I'm confused. Let me look at the graph output shapes again:
    # conv2d.default -> shape=[2, 96, 1, 1]  <- this