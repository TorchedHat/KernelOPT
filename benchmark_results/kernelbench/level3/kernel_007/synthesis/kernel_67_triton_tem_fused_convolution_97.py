import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    WITH_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)
        
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_h & valid_w
        
        x_ptrs = (X_ptr 
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        valid_n = offs_n[None, :] < OUT_C
        valid_k2 = offs_k[:, None] < TOTAL_K
        mask_w = valid_k2 & valid_n
        
        w_ptrs = (W_ptr 
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    if WITH_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
    
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    
    y_ptrs = (Y_ptr 
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # offs indexes into (BATCH, C, OUT_H, OUT_W)
    out_w = offs % OUT_W
    out_h = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    b = offs // (OUT_W * OUT_H * C)
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = out_h * STRIDE_H + kh - PAD_H
            iw = out_w * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + b * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(Y_ptr + offs, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = W_ptr + offs_k[:, None] * N + offs_n[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    X_ptr, Y_ptr,
    N,
    BATCH, C, IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    c = offs % C
    b = offs // C
    
    total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = IN_H * IN_W
    
    for h in range(IN_H):
        for w in range(IN_W):
            x_ptrs = X_ptr + b * (C * IN_H * IN_W) + c * (IN_H * IN_W) + h * IN_W + w
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            total = total + val
    
    result = total / count
    tl.store(Y_ptr + offs, result, mask=mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), relu=False):
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1
    
    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    
    M = N * OH * OW
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    
    conv2d_relu_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        relu,
        BATCH=N, IN_C=C_in, IN_H=H, IN_W=W, OUT_C=C_out, OUT_H=OH, OUT_W=OW,
        KERNEL_H=KH, KERNEL_W=KW,
    )
    return y


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1
    
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x, y,
        total,
        N, C, H, W, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        N=total,
    )
    return y


def triton_adaptive_avg_pool(x):
    N, C, H, W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total = N * C
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    adaptive_avg_pool_kernel[grid](
        x, y,
        total,
        N, C, H, W,
        N=total,
    )
    return y


def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    linear_kernel[grid](
        x, weight, bias, y,
        M, N, K,
    )
    return y


def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    # branch1x1
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), relu=True)
    
    # branch3x3
    r3 = triton_conv2d(x, w_r3, b_r3, stride=(1,1), padding=(0,0), relu=True)
    b3 = triton_conv2d(r3, w_3x3, b_3x3, stride=(1,1), padding=(1,1), relu=True)
    
    # branch5x5
    r5 = triton_conv2d(x, w_r5, b_r5, stride=(1,1), padding=(0,0), relu=True)
    b5 = triton_conv2d(r5, w_5x5, b_5x5, stride=(1,1), padding=(2,2), relu=True)
    
    # branch_pool: maxpool then conv
    mp = triton_maxpool2d(x, kernel_size=3, stride=1, padding=1)
    bp = triton_conv2d(mp, w_pool, b_pool, stride=(1,1), padding=(0,0), relu=True)
    
    return torch.cat([b1, b3, b5, bp], dim=1)


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
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
    x,
):
    # conv1: 3->64, 7x7, stride=2, pad=3
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 64->64, 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), relu=True)
    # conv3: 64->192, 3x3, pad=1
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception3a: in=192, out_1x1=64, r3=96, out3=128, r5=16, out5=32, pool=32
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )
    
    # inception3b: in=256, out_1x1=128, r3=128, out3=192, r5=32, out5=96, pool=64
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )
    
    # maxpool3: 3x3, stride=2, pad=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception4a: in=480
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
        p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    )
    
    # inception4b: in=512
    x = inception_module(x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
        p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    )
    
    # inception4c: in=512
    x = inception_module(x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
        p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
        p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    )
    
    # inception4d: in=512
    x = inception_module(x,