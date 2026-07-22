import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Fused BatchNorm + ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C,
    eps,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # channel index: layout is (N, C, H, W) so channel = (offset // HW) % C
    # But we pass numel = N*C*H*W and need channel index
    # We'll compute channel from offset
    # Actually we need HW = numel / (N*C) but we don't have N here
    # Instead pass C and compute: channel = (offset // 1) % C won't work for NCHW
    # For NCHW: offset = n*C*HW + c*HW + h*W + w
    # channel = (offset // HW) % C
    # We need HW. Let's pass it as a parameter.
    # Rewrite: pass HW separately
    # For now this kernel signature needs HW - let's restructure

    # Actually let's just use channel = (offset // HW) % C
    # We'll add HW parameter
    tl.store(out_ptr + offsets, x, mask=mask)


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
def fused_bn_relu_kernel_v2(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, HW,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    c = (offsets // HW) % (numel // HW)
    # Actually c = (offsets // HW) but we need to mod by C
    # numel = N*C*HW, so C = numel // (N*HW)
    # Let's just pass C
    c_idx = (offsets // HW)
    # This gives n*C + c, we need just c
    # We need C to mod
    # Let's restructure to pass C

    tl.store(out_ptr + offsets, x, mask=mask)


# Proper implementation
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
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = (offsets // HW) % C

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    eps = 1e-5
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b

    if apply_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Fused Sigmoid pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Adaptive Average Pool 2D kernel (global average pooling to 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    nc = pid
    n = nc // C
    c = nc % C

    if n >= N:
        return

    # Sum over HW
    acc = tl.zeros([1], dtype=tl.float32)
    base = n * C * HW + c * HW

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        x = tl.load(x_ptr + base + hw_offsets, mask=hw_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)

    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# ============================================================
# Fused Linear (GEMM) kernel: y = x @ W^T + b
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # W is (N, K), we want W^T which is (K, N)
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(x, w, acc)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Conv2d GEMM kernel (implicit GEMM for 1x1 convolutions)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """1x1 convolution as GEMM: x(M, K) @ w^T(K, N) -> out(M, N)
    where M = N_batch * H * W, K = C_in, N = C_out
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # w is (N, K) -> load as (K, N) for dot
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(x, w, acc)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 1}, num_warps=2),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Depthwise 3x3 convolution"""
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)

    M = N * OUT_H * OUT_W
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_m = offs_m < M
    mask_c = c < C

    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_H * OUT_W)

    acc = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W

            valid = mask_m & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)

            x_idx = n * C * IN_H * IN_W + c[None, :] * IN_H * IN_W + ih[:, None] * IN_W + iw[:, None]
            x_val = tl.load(x_ptr + x_idx, mask=valid[:, None] & mask_c[None, :], other=0.0).to(tl.float32)

            w_idx = c * 9 + kh * 3 + kw
            w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0).to(tl.float32)

            acc += x_val * w_val[None, :]

    out_idx = n * C * OUT_H * OUT_W + c[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    tl.store(out_ptr + out_idx, acc, mask=mask_m[:, None] & mask_c[None, :])


# ============================================================
# Conv3x3 (standard) kernel using implicit GEMM
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard 3x3 convolution as implicit GEMM"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    TOTAL_K = IN_C * 9  # 3x3 kernel

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    bn = offs_m // (OUT_H * OUT_W)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        ic = offs_k % IN_C
        kw = (offs_k // IN_C) % 3
        kh = offs_k // (IN_C * 3)

        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (offs_m[:, None] < M) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_k[None, :] < TOTAL_K)

        x_ptrs = x_ptr + bn[:, None] * IN_C * IN_H * IN_W + ic[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)

        # w: (OUT_C, IN_C, 3, 3) -> w[oc, ic, kh, kw]
        w_ptrs = w_ptr + offs_n[None, :] * IN_C * 9 + ic[:, None] * 9 + kh[:, None] * 3 + kw[:, None]
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(x_tile, w_tile, acc)

    out_ptrs = out_ptr + bn[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Helper functions
# ============================================================
def triton_conv1x1(x, w):
    """x: (N, C_in, H, W), w: (C_out, C_in, 1, 1) -> (N, C_out, H, W)"""
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    M = N * H * W
    K = C_in

    # Reshape to (M, K)
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_2d = w.view(C_out, K)  # (N_out, K)

    out_2d = torch.empty(M, C_out, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_kernel[grid](x_2d, w_2d, out_2d, M, C_out, K)

    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def triton_conv3x3(x, w, stride=1, padding=1):
    """x: (N, C_in, H, W), w: (C_out, C_in, 3, 3) -> (N, C_out, OH, OW)"""
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    OH = (H + 2 * padding - 3) // stride + 1
    OW = (W + 2 * padding - 3) // stride + 1
    M = N * OH * OW

    out = torch.empty(N, C_out, OH, OW, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv3x3_kernel[grid](
        x, w, out,
        N, C_in, H, W, C_out, OH, OW,
    )
    return out


def triton_depthwise_conv3x3(x, w, stride=1, padding=1):
    """x: (N, C, H, W), w: (C, 1, 3, 3) -> (N, C, OH, OW)"""
    N, C, H, W = x.shape
    OH = (H + 2 * padding - 3) // stride + 1
    OW = (W + 2 * padding - 3) // stride + 1
    M = N * OH * OW

    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)

    # w: (C, 1, 3, 3) -> (C, 9)
    w_flat = w.view(C, 9)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C, meta['BLOCK_C']))
    depthwise_conv3x3_kernel[grid](
        x, w_flat, out,
        N, C, H, W, OH, OW,
    )
    return out


def triton_bn_relu(x, mean, var, weight, bias, apply_relu=True):
    """Fused batch norm + optional relu"""
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW,
        apply_relu=apply_relu
    )
    return out


def triton_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_adaptive_avg_pool2d(x):
    """Global average pooling to 1x1"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)

    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](
        x, out.view(N * C),
    )
    return out


def triton_linear(x, w, b):
    """x: (M, K), w: (N, K), b: (N,) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, b, out, M, N, K)
    return out


# ============================================================
# MBConv block helper
# ============================================================
def mbconv_block(
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
):
    # Expansion
    x = triton_conv1x1(x, expand_w)
    x = triton_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, apply_relu=True)

    # Depthwise
    x = triton_depthwise_conv3x3(x, dw_w, stride=stride, padding=1)
    x = triton_bn_relu(