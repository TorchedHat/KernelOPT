import torch
import triton
import triton.language as tl
import math

# Due to the enormous complexity of DenseNet-121 with hundreds of layers,
# I'll implement the key fused operations in Triton and structure the forward pass.

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
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
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
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


import torch
import triton
import triton.language as tl

# Fused BatchNorm + ReLU kernel
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
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
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
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
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


# Implicit GEMM convolution kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H, W,
    OC, OH, OW,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    M, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    bn = offs_m // (OW * OH)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (ic, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ic_idx = offs_k // (KW * KH)
        
        ih = oh[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw_idx[None, :] - pad_w
        
        valid_x = (bn[:, None] < N) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & (offs_k[None, :] < K)
        
        x_ptrs = x_ptr + bn[:, None] * stride_xn + ic_idx[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OC)
        w_ptrs = w_ptr + offs_n[None, :] * stride_woc + ic_idx[:, None] * stride_wic + kh_idx[:, None] * stride_wkh + kw_idx[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OC)
    out_ptrs = out_ptr + bn[:, None] * (OC * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, C, H, W = x.shape
    OC, IC, KH, KW = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    OH = (H + 2*pad_h - KH) // stride_h + 1
    OW = (W + 2*pad_w - KW) // stride_w + 1
    
    out = torch.empty(N, OC, OH, OW, device=x.device, dtype=x.dtype)
    
    M = N * OH * OW
    K = IC * KH * KW
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))
    
    conv2d_kernel[grid](
        x, w, out,
        N, C, H, W,
        OC, OH, OW,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
    )
    return out


def triton_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


def triton_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


# Max pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W, OH, OW,
    KH, KW, stride_h, stride_w, pad_h, pad_w,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    OH = (H + 2*padding - KH) // stride + 1
    OW = (W + 2*padding - KW) // stride + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, H, W, OH, OW,
        KH, KW, stride, stride, padding, padding,
        numel,
    )
    return out


# Avg pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W, OH, OW,
    KH, KW, stride_h, stride_w,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh
            iw = ow * stride_w + kw
            valid = mask & (ih < H) & (iw < W)
            x_idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + val
    
    tl.store(out_ptr + offsets, acc / count, mask=mask)


def triton_avg_pool2d(x, kernel_size=2, stride=2):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W, OH, OW,
        KH, KW, stride, stride,
        numel,
    )
    return out


# Adaptive avg pool kernel (global average pooling)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['numel', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = offsets % C
    n = offsets // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for hw in range(HW):
        x_idx = n * (C * HW) + c * HW + hw
        val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        acc = acc + val
    
    tl.store(out_ptr + offsets, acc / HW, mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, HW,
        numel,
    )
    return out


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def triton_linear(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, b, out, M, N, K)
    return out


def kernel_function(*inputs):
    idx = 0
    # Unpack all inputs
    p_features_0_weight = inputs[idx]; idx+=1
    p_features_1_weight = inputs[idx]; idx+=1
    p_features_1_bias = inputs[idx]; idx+=1
    
    # Dense block 0 (6 layers)
    p_db0_l0_bn_w = inputs[idx]; idx+=1
    p_db0_l0_bn_b = inputs[idx]; idx+=1
    p_db0_l0_conv_w = inputs[idx]; idx+=1
    p_db0_l1_bn_w = inputs[idx]; idx+=1
    p_db0_l1_bn_b = inputs[idx]; idx+=1
    p_db0_l1_conv_w = inputs[idx]; idx+=1
    p_db0_l2_bn_w = inputs[idx]; idx+=1
    p_db0_l2_bn_b = inputs[idx]; idx+=1
    p_db0_l2_conv_w = inputs[idx]; idx+=1
    p_db0_l3_bn_w = inputs[idx]; idx+=1
    p_db0_l3_bn_b = inputs[idx]; idx+=1
    p_db0_l3_conv_w = inputs[idx]; idx+=1
    p_db0_l4_bn_w = inputs[idx]; idx+=1
    p_db0_l4_bn_b = inputs[idx]; idx+=1
    p_db0_l4_conv_w = inputs[idx]; idx+=1
    p_db0_l5_bn_w = inputs[idx]; idx+=1
    p_db0_l5_bn_b = inputs[idx]; idx+=1
    p_db0_l5_conv_w = inputs[idx]; idx+=1
    
    # Transition 0
    p_tr0_bn_w = inputs[idx]; idx+=1
    p_tr0_bn_b = inputs[idx]; idx+=1
    p_tr0_conv_w = inputs[idx]; idx+=1
    
    # Dense block 1 (12 layers)
    p_db1_weights = []
    for i in range(12):
        p_db1_weights.append((inputs[idx], inputs[idx+1], inputs[idx+2]))
        idx += 3
    
    # Transition 1
    p_tr1_bn_w = inputs[idx]; idx+=1
    p_tr1_bn_b = inputs[idx]; idx+=1
    p_tr1_conv_w = inputs[idx]; idx+=1
    
    # Dense block 2 (24 layers)
    p_db2_weights = []
    for i in range(24):
        p_db2_weights.append((inputs[idx], inputs[idx+1], inputs[idx+2]))
        idx += 3
    
    # Transition 2
    p_tr2_bn_w = inputs[idx]; idx+=1
    p_tr2_bn_b = inputs[idx]; idx+=1
    p_tr2_conv_w = inputs[idx]; idx+=1
    
    # Dense block 3 (16 layers)
    p_db3_weights = []
    for i in range(16):
        p_db3_weights.append((inputs[idx], inputs[idx+1], inputs[idx+2]))
        idx += 3
    
    # Final BN
    p_final_bn_w = inputs[idx]; idx+=1
    p_final_bn_b = inputs[idx]; idx+=1
    
    # Classifier
    p_cls_w = inputs[idx]; idx+=1
    p_cls_b = inputs[idx]; idx+=1
    
    # Running stats
    b_features_1_rm = inputs[idx]; idx+=1
    b_features_1_rv = inputs[idx]; idx+=1
    
    # DB0 running stats
    b_db0_rm = []; b_db0_rv = []
    for i in range(6):
        b_db0_rm.append(inputs[idx]); idx+=1
        b_db0_rv.append(inputs[idx]); idx+=1
    
    # TR0 running stats
    b_tr0_rm = inputs[idx]; idx+=1
    b_tr0_rv = inputs[idx]; idx+=1
    
    # DB1 running stats
    b_db1_rm = []; b_db1_rv = []
    for i in range(12):
        b_db1_rm.append(inputs[idx]); idx+=1
        b_db1_rv.append(inputs[idx]); idx+=1
    
    # TR1 running stats
    b_tr1_rm = inputs[idx]; idx+=1
    b_tr1_rv = inputs[idx]; idx+=1
    
    # DB2 running stats
    b_db2_rm = []; b_db2_rv = []
    for i in range(24):
        b_db2_rm.append(inputs[idx]); idx+=1
        b_db2_rv.append(inputs[idx]); idx+=1
    
    # TR2 running stats
    b_tr2_rm = inputs[idx]; idx+=1
    b_tr2_rv = inputs[idx]; idx+=1
    
    # DB3 running stats
    b_db3_rm = []; b_db3_rv = []
    for i in range(16):
        b_db3_rm.append(inputs[idx]); idx+=1
        b_db3_rv.append(inputs[idx]); idx+=1
    
    # Final BN running stats
    b_final_bn_rm = inputs[idx]; idx+=1
    b_final_bn_rv = inputs[idx]; idx+=1
    
    # Input
    x = inputs[idx]; idx+=1
    
    # Forward pass
    # Initial conv: 3->64, 7x7, stride=2, pad=3
    x = triton_conv2d(x, p_features_0_weight, stride=(2,2), padding=(3,3))
    x = triton_bn_relu(x, p_features_1_weight, p_features_1_bias, b_features_1_rm, b_features_1_rv)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Dense block 0
    db0_bn_weights = [p_db0_l0_bn_w, p_db0_l1_bn_w, p_db0_l2_bn_w, p_db0_l3_bn_w, p_db0_l4_bn_w, p_db0_l5_bn_w]
    db0_bn_biases = [p_db0_l0_bn_