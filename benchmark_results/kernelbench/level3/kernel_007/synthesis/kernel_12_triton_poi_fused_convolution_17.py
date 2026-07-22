import torch
import triton
import triton.language as tl

# Fused bias add + relu kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# Max pool 3x3 stride 2 padding 1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_3x3_s2_p1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    ih_base = oh * 2 - 1
    iw_base = ow * 2 - 1
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = ih_base + kh
            iw = iw_base + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


# Max pool 3x3 stride 1 padding 1 (used in inception branches)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_3x3_s1_p1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    ih_base = oh - 1
    iw_base = ow - 1
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = ih_base + kh
            iw = iw_base + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


# Adaptive avg pool 7x7 -> 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    c = offsets % C
    n = offsets // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IH * IW
    
    for h in range(IH):
        for w in range(IW):
            idx = n * C * IH * IW + c * IH * IW + h * IW + w
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val
    
    avg = acc / count
    tl.store(out_ptr + offsets, avg, mask=mask)


# Linear (matrix multiply) kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
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
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)
    
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# Implicit GEMM convolution kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2),
    ],
    key=['M', 'N', 'K', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DO_RELU: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n_idx = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ic, kh, kw)
        kw = offs_k % KERNEL_W
        kh = (offs_k // KERNEL_W) % KERNEL_H
        ic = offs_k // (KERNEL_W * KERNEL_H)
        
        # Input coordinates
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_x = (
            (offs_m[:, None] < M) &
            (n_idx[:, None] < N_batch) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )
        
        x_idx = n_idx[:, None] * IN_C * IN_H * IN_W + ic[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)
        
        # Weight: [OUT_C, IN_C, KH, KW] -> index [oc, ic, kh, kw]
        # w_idx for each (k, n): W[offs_n, ic, kh, kw]
        w_idx = offs_n[None, :] * IN_C * KERNEL_H * KERNEL_W + ic[:, None] * KERNEL_H * KERNEL_W + kh[:, None] * KERNEL_W + kw[:, None]
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    if HAS_BIAS:
        b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + b[None, :]
    
    if DO_RELU:
        acc = tl.where(acc > 0, acc, 0.0)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    # Store in NCHW format: Y[n, oc, oh, ow]
    out_idx = n_idx[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), relu=False):
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    SH, SW = stride
    PH, PW = padding
    
    OUT_H = (IN_H + 2*PH - KERNEL_H) // SH + 1
    OUT_W = (IN_W + 2*PW - KERNEL_W) // SW + 1
    
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KERNEL_H * KERNEL_W
    
    y = torch.empty(N_batch, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, weight, bias if bias is not None else x,
        N_batch, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        SH, SW, PH, PW,
        M, N, K,
        KERNEL_H, KERNEL_W,
        HAS_BIAS=(bias is not None),
        DO_RELU=relu,
    )
    return y


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IH, IW = x.shape
    OH = (IH + 2*padding - kernel_size) // stride + 1
    OW = (IW + 2*padding - kernel_size) // stride + 1
    
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    
    if stride == 2 and padding == 1 and kernel_size == 3:
        grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
        max_pool2d_3x3_s2_p1_kernel[grid](x, out, N, C, IH, IW, OH, OW)
    else:
        # stride=1, padding=1
        grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
        max_pool2d_3x3_s1_p1_kernel[grid](x, out, N, C, IH, IW, OH, OW)
    return out


def triton_adaptive_avg_pool2d(x):
    N, C, IH, IW = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](x, out, N, C, IH, IW)
    return out


def triton_linear(x, weight, bias):
    # x: [M, K], weight: [N, K], bias: [N]
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    # weight is [N, K], we need [K, N] for matmul
    w_t = weight.t().contiguous()
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](x, w_t, bias, out, M, N, K)
    return out


def inception_module(x, 
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1: 1x1 conv + relu
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), relu=True)
    
    # Branch 2: 1x1 reduce + relu, then 3x3 + relu
    branch2 = triton_conv2d(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), relu=True)
    branch2 = triton_conv2d(branch2, w_3x3, b_3x3, stride=(1,1), padding=(1,1), relu=True)
    
    # Branch 3: 1x1 reduce + relu, then 5x5 + relu
    branch3 = triton_conv2d(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), relu=True)
    branch3 = triton_conv2d(branch3, w_5x5, b_5x5, stride=(1,1), padding=(2,2), relu=True)
    
    # Branch 4: max pool 3x3 s1 p1, then 1x1 + relu
    branch4 = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch4 = triton_conv2d(branch4, w_pool, b_pool, stride=(1,1), padding=(0,0), relu=True)
    
    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,