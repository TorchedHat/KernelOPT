import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, HW,
        eps,
    )
    return out


def kernel_function(
    p_features_0_weight, p_features_1_weight, p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight, p_dense_blocks_0_layers_0_0_bias, p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight, p_dense_blocks_0_layers_1_0_bias, p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight, p_dense_blocks_0_layers_2_0_bias, p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight, p_dense_blocks_0_layers_3_0_bias, p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight, p_dense_blocks_0_layers_4_0_bias, p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight, p_dense_blocks_0_layers_5_0_bias, p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight, p_transition_layers_0_transition_0_bias, p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight, p_dense_blocks_1_layers_0_0_bias, p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight, p_dense_blocks_1_layers_1_0_bias, p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight, p_dense_blocks_1_layers_2_0_bias, p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight, p_dense_blocks_1_layers_3_0_bias, p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight, p_dense_blocks_1_layers_4_0_bias, p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight, p_dense_blocks_1_layers_5_0_bias, p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight, p_dense_blocks_1_layers_6_0_bias, p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight, p_dense_blocks_1_layers_7_0_bias, p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight, p_dense_blocks_1_layers_8_0_bias, p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight, p_dense_blocks_1_layers_9_0_bias, p_dense_blocks_1_layers_9_2_weight,
    p_dense_blocks_1_layers_10_0_weight, p_dense_blocks_1_layers_10_0_bias, p_dense_blocks_1_layers_10_2_weight,
    p_dense_blocks_1_layers_11_0_weight, p_dense_blocks_1_layers_11_0_bias, p_dense_blocks_1_layers_11_2_weight,
    p_transition_layers_1_transition_0_weight, p_transition_layers_1_transition_0_bias, p_transition_layers_1_transition_2_weight,
    p_dense_blocks_2_layers_0_0_weight, p_dense_blocks_2_layers_0_0_bias, p_dense_blocks_2_layers_0_2_weight,
    p_dense_blocks_2_layers_1_0_weight, p_dense_blocks_2_layers_1_0_bias, p_dense_blocks_2_layers_1_2_weight,
    p_dense_blocks_2_layers_2_0_weight, p_dense_blocks_2_layers_2_0_bias, p_dense_blocks_2_layers_2_2_weight,
    p_dense_blocks_2_layers_3_0_weight, p_dense_blocks_2_layers_3_0_bias, p_dense_blocks_2_layers_3_2_weight,
    p_dense_blocks_2_layers_4_0_weight, p_dense_blocks_2_layers_4_0_bias, p_dense_blocks_2_layers_4_2_weight,
    p_dense_blocks_2_layers_5_0_weight, p_dense_blocks_2_layers_5_0_bias, p_dense_blocks_2_layers_5_2_weight,
    p_dense_blocks_2_layers_6_0_weight, p_dense_blocks_2_layers_6_0_bias, p_dense_blocks_2_layers_6_2_weight,
    p_dense_blocks_2_layers_7_0_weight, p_dense_blocks_2_layers_7_0_bias, p_dense_blocks_2_layers_7_2_weight,
    p_dense_blocks_2_layers_8_0_weight, p_dense_blocks_2_layers_8_0_bias, p_dense_blocks_2_layers_8_2_weight,
    p_dense_blocks_2_layers_9_0_weight, p_dense_blocks_2_layers_9_0_bias, p_dense_blocks_2_layers_9_2_weight,
    p_dense_blocks_2_layers_10_0_weight, p_dense_blocks_2_layers_10_0_bias, p_dense_blocks_2_layers_10_2_weight,
    p_dense_blocks_2_layers_11_0_weight, p_dense_blocks_2_layers_11_0_bias, p_dense_blocks_2_layers_11_2_weight,
    p_dense_blocks_2_layers_12_0_weight, p_dense_blocks_2_layers_12_0_bias, p_dense_blocks_2_layers_12_2_weight,
    p_dense_blocks_2_layers_13_0_weight, p_dense_blocks_2_layers_13_0_bias, p_dense_blocks_2_layers_13_2_weight,
    p_dense_blocks_2_layers_14_0_weight, p_dense_blocks_2_layers_14_0_bias, p_dense_blocks_2_layers_14_2_weight,
    p_dense_blocks_2_layers_15_0_weight, p_dense_blocks_2_layers_15_0_bias, p_dense_blocks_2_layers_15_2_weight,
    p_dense_blocks_2_layers_16_0_weight, p_dense_blocks_2_layers_16_0_bias, p_dense_blocks_2_layers_16_2_weight,
    p_dense_blocks_2_layers_17_0_weight, p_dense_blocks_2_layers_17_0_bias, p_dense_blocks_2_layers_17_2_weight,
    p_dense_blocks_2_layers_18_0_weight, p_dense_blocks_2_layers_18_0_bias, p_dense_blocks_2_layers_18_2_weight,
    p_dense_blocks_2_layers_19_0_weight, p_dense_blocks_2_layers_19_0_bias, p_dense_blocks_2_layers_19_2_weight,
    p_dense_blocks_2_layers_20_0_weight, p_dense_blocks_2_layers_20_0_bias, p_dense_blocks_2_layers_20_2_weight,
    p_dense_blocks_2_layers_21_0_weight, p_dense_blocks_2_layers_21_0_bias, p_dense_blocks_2_layers_21_2_weight,
    p_dense_blocks_2_layers_22_0_weight, p_dense_blocks_2_layers_22_0_bias, p_dense_blocks_2_layers_22_2_weight,
    p_dense_blocks_2_layers_23_0_weight, p_dense_blocks_2_layers_23_0_bias, p_dense_blocks_2_layers_23_2_weight,
    p_dense_blocks_2_layers_24_0_weight, p_dense_blocks_2_layers_24_0_bias, p_dense_blocks_2_layers_24_2_weight,
    p_dense_blocks_2_layers_25_0_weight, p_dense_blocks_2_layers_25_0_bias, p_dense_blocks_2_layers_25_2_weight,
    p_dense_blocks_2_layers_26_0_weight, p_dense_blocks_2_layers_26_0_bias, p_dense_blocks_2_layers_26_2_weight,
    p_dense_blocks_2_layers_27_0_weight, p_dense_blocks_2_layers_27_0_bias, p_dense_blocks_2_layers_27_2_weight,
    p_dense_blocks_2_layers_28_0_weight, p_dense_blocks_2_layers_28_0_bias, p_dense_blocks_2_layers_28_2_weight,
    p_dense_blocks_2_layers_29_0_weight, p_dense_blocks_2_layers_29_0_bias, p_dense_blocks_2_layers_29_2_weight,
    p_dense_blocks_2_layers_30_0_weight, p_dense_blocks_2_layers_30_0_bias, p_dense_blocks_2_layers_30_2_weight,
    p_dense_blocks_2_layers_31_0_weight, p_dense_blocks_2_layers_31_0_bias, p_dense_blocks_2_layers_31_2_weight,
    p_dense_blocks_2_layers_32_0_weight, p_dense_blocks_2_layers_32_0_bias, p_dense_blocks_2_layers_32_2_weight,
    p_dense_blocks_2_layers_33_0_weight, p_dense_blocks_2_layers_33_0_bias, p_dense_blocks_2_layers_33_2_weight,
    p_dense_blocks_2_layers_34_0_weight, p_dense_blocks_2_layers_34_0_bias, p_dense_blocks_2_layers_34_2_weight,
    p_dense_blocks_2_layers_35_0_weight, p_dense_blocks_2_layers_35_0_bias, p_dense_blocks_2_layers_35_2_weight,
    p_dense_blocks_2_layers_36_0_weight, p_dense_blocks_2_layers_36_0_bias, p_dense_blocks_2_layers_36_2_weight,
    p_dense_blocks_2_layers_37_0_weight, p_dense_blocks_2_layers_37_0_bias, p_dense_blocks_2_layers_37_2_weight,
    p_dense_blocks_2_layers_38_0_weight, p_dense_blocks_2_layers_38_0_bias, p_dense_blocks_2_layers_38_2_weight,
    p_dense_blocks_2_layers_39_0_weight, p_dense_blocks_2_layers_39_0_bias, p_dense_blocks_2_layers_39_2_weight,
    p_dense_blocks_2_layers_40_0_weight, p_dense_blocks_2_layers_40_0_bias, p_dense_blocks_2_layers_40_2_weight,
    p_dense_blocks_2_layers_41_0_weight, p_dense_blocks_2_layers_41_0_bias, p_dense_blocks_2_layers_41_2_weight,
    p_dense_blocks_2_layers_42_0_weight, p_dense_blocks_2_layers_42_0_bias, p_dense_blocks_2_layers_42_2_weight,
    p_dense_blocks_2_layers_43_0_weight, p_dense_blocks_2_layers_43_0_bias, p_dense_blocks_2_layers_43_2_weight,
    p_dense_blocks_2_layers_44_0_weight, p_dense_blocks_2_layers_44_0_bias, p_dense_blocks_2_layers_44_2_weight,
    p_dense_blocks_2_layers_45_0_weight, p_dense_blocks_2_layers_45_0_bias, p_dense_blocks_2_layers_45_2_weight,
    p_dense_blocks_2_layers_46_0_weight, p_dense_blocks_2_layers_46_0_bias, p_dense_blocks_2_layers_46_2_weight,
    p_dense_blocks_2_layers_47_0_weight, p_dense_blocks_2_layers_47_0_bias, p_dense_blocks_2_layers_47_2_weight,
    p_transition_layers_2_transition_0_weight, p_transition_layers_2_transition_0_bias, p_transition_layers_2_transition_2_weight,
    p_dense_blocks_3_layers_0_0_weight, p_dense_blocks_3_layers_0_0_bias, p_dense_blocks_3_layers_0_2_weight,
):
    pass


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
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

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
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

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'out_C', 'in_C'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple row-parallel linear for small M
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    
    for col in range(N):
        acc = tl.zeros((), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE):
            offs = k + tl.arange(0, BLOCK_SIZE)
            mask = offs < K
            x_val = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0)
            w_val = tl.load(w_ptr + col * K + offs, mask=mask, other=0.0)
            acc += tl.sum(x_val * w_val, axis=0)
        b_val = tl.load(b_ptr + col)
        tl.store(out_ptr + row * N + col, acc + b_val)


def triton_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out, numel, C, HW, eps,
    )
    return out


def triton_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out, numel, C, HW, eps,
    )
    return out


def dense_layer(x, bn_w, bn_b, bn_rm, bn_rv, conv_w, dropout_p=0.0):
    x = triton_bn_relu(x, bn_w, bn_b, bn_rm, bn_rv)
    x = torch.nn.functional.conv2d(x, conv_w, padding=1)
    return x


def dense_block(x, layer_params):
    features = [x]
    for (bn_w, bn_b, bn_rm, bn_rv, conv_w) in layer_params:
        new_feat = dense_layer(x, bn_w, bn_b, bn_rm, bn_rv, conv_w)
        features.append(new_feat)
        x = torch.cat(features, dim=1)
    return x


def transition_layer(x, bn_w, bn_b, bn_rm, bn_rv, conv_w):
    x = triton_bn_relu(x, bn_w, bn_b, bn_rm, bn_rv)
    x = torch.nn.functional.conv2d(x, conv_w)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    return x


def kernel_function(*args):
    idx = 0
    def next_arg():
        nonlocal idx
        v = args[idx]
        idx += 1
        return v

    p_features_0_weight = next_arg()
    p_features_1_weight = next_arg()
    p_features_1_bias = next_arg()

    # Dense block 0: 6 layers
    db0_params = []
    for i in range(6):
        bn_w = next_arg(); bn_b = next_arg(); conv_w = next_arg()
        db0_params.append((bn_w, bn_b, None, None, conv_w))

    # Transition 0
    t0_bn_w = next_arg(); t0_bn_b = next_arg(); t0_conv_w = next_arg()

    # Dense block 1: 12 layers
    db1_params = []
    for i in range(12):
        bn_w = next_arg(); bn_b = next_arg(); conv_w = next_arg()
        db1_params.append((bn_w, bn_b, None, None, conv_w))

    # Transition 1
    t1_bn_w = next_arg(); t1_bn_b = next_arg(); t1_conv_w = next_arg()

    # Dense block 2: 48 layers
    db2_params = []
    for i in range(48):
        bn_w = next_arg(); bn_b = next_arg(); conv_w = next_arg()
        db2_params.append((bn_w, bn_b, None, None, conv_w))

    # Transition 2
    t2_bn_w = next_arg(); t2_bn_b = next_arg(); t2_conv_w = next_arg()

    # Dense block 3: 32 layers
    db3_params = []
    for i in range(32):
        bn_w = next_arg(); bn_b = next_arg(); conv_w = next_arg()
        db3_params.append((bn_w, bn_b, None, None, conv_w))

    p_final_bn_weight = next_arg()
    p_final_bn_bias = next_arg()
    p_classifier_weight = next_arg()
    p_classifier_bias = next_arg()

    # Running stats
    b_features_1_running_mean = next_arg()
    b_features_1_running_var = next_arg()

    db0_rm = []; db0_rv = []
    for i in range(6):
        db0_rm.append(next_arg()); db0_rv.append(next_arg())

    t0_rm = next_arg(); t0_rv = next_arg()

    db1_rm = []; db1_rv = []
    for i in range(12):
        db1_rm.append(next_arg()); db1_rv.append(next_arg())

    t1_rm = next_arg(); t1_rv = next_arg()

    db2_rm = []; db2_rv = []
    for i in range(48):
        db2_rm.append(next_arg()); db2_rv.append(next_arg())

    t2_rm = next_arg(); t2_rv = next_arg()

    db3_rm = []; db3_rv = []
    for i in range(32):
        db3_rm.append(next_arg()); db3_rv.append(next_arg())

    b_final_bn_running_mean = next_arg()
    b_final_bn_running_var = next_arg()

    x = next_arg()

    # Initial features
    x = torch.nn.functional.conv2d(x, p_features_0_weight, stride=2, padding=3)
    x = triton_bn_relu(x, p_features_1_weight, p_features_1_bias,
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # Dense block 0
    features = [x]
    for i, (bn_w, bn_b, _, _, conv_w) in enumerate(db0_params):
        inp = triton_bn_relu(x, bn_w, bn_b, db0_rm[i], db0_rv[i])
        new_feat = torch.nn.functional.conv2d(inp, conv_w, padding=1)
        features.append(new_feat)
        x = torch.cat(features, dim=1)

    # Transition 0
    x = triton_bn_relu(x, t0_bn_w, t0_bn_b, t0_rm, t0_rv)
    x = torch.nn.functional.conv2d(x, t0_conv_w)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

    # Dense block 1
    features = [x]
    for i, (bn_w, bn_b, _, _, conv_w) in enumerate(db1_params):
        inp = triton_bn_relu(x, bn_w, bn_b, db1_rm[i], db1_rv[i])
        new_feat = torch.nn.functional.conv2d(inp, conv_w, padding=1)
        features.append(new_feat)
        x = torch.cat(features, dim=1)

    # Transition 1
    x = triton_bn_relu(x, t1_bn_w, t1_bn_b, t1_rm, t1_rv)
    x = torch.nn.functional.conv2d(x, t1_conv_w)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

    # Dense block 2
    features = [x]
    for i, (bn_w, bn_b, _, _, conv_w) in enumerate(db2_params):
        inp = triton_bn_relu(x, bn_w, bn_b, db2_rm[i], db2_rv[i])
        new_feat = torch.nn.functional.conv2d(inp, conv_w, padding=1)
        features.append(new_feat)
        x = torch.cat(features, dim=1)

    # Transition 2
    x = triton_bn_relu(x, t2_bn_w, t2_bn_b, t2_rm, t2_rv)
    x = torch.nn.functional.conv2d(x, t2_conv_w)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

    # Dense block 3