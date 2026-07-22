import torch
import triton
import triton.language as tl
import math

# ============================================================
# Utility: ceiling division
# ============================================================
def cdiv(a, b):
    return (a + b - 1) // b

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM style)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W', 'USE_RELU'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    USE_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)
        
        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw
        
        # X is NCHW: stride = (IN_C*IN_H*IN_W, IN_H*IN_W, IN_W, 1)
        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=mask_x, other=0.0)
        
        # W is (OUT_C, IN_C, KERNEL_H, KERNEL_W): stride = (IN_C*KH*KW, KH*KW, KW, 1)
        # We need W[out_c, in_c, kh, kw] for each (k, n)
        # w_idx[k, n] = offs_n[n] * (IN_C*KH*KW) + in_c[k] * (KH*KW) + kh[k] * KW + kw[k]
        w_idx = offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + in_c[:, None] * (KERNEL_H * KERNEL_W) + kh[:, None] * KERNEL_W + kw[:, None]
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc += tl.dot(x_tile, w_tile, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    # ReLU
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)
    
    # Store: output is NCHW
    # Y[batch, out_c, out_h, out_w] = Y[batch * OUT_C*OUT_H*OUT_W + out_c * OUT_H*OUT_W + out_h * OUT_W + out_w]
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=mask_out)

def conv2d_bias_relu(x, w, b, stride_h, stride_w, pad_h, pad_w, use_relu=True):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w.shape
    OUT_H = (IN_H + 2 * pad_h - KERNEL_H) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KERNEL_W) // stride_w + 1
    
    y = torch.empty(BATCH, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))
    
    conv2d_bias_relu_kernel[grid](
        x, w, b, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        stride_h, stride_w,
        pad_h, pad_w,
        use_relu,
    )
    return y


# ============================================================
# MaxPool2d kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['BATCH', 'C', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W', 'KH', 'KW', 'STRIDE_H', 'STRIDE_W'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH: tl.constexpr, C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    TOTAL = BATCH * C * OUT_H * OUT_W
    mask = offs < TOTAL
    
    # Decode: NCHW layout
    out_w = offs % OUT_W
    tmp = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    # Compute max over kernel window
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * STRIDE_H + kh - PAD_H
            iw = out_w * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(Y_ptr + offs, max_val, mask=mask)


def maxpool2d(x, kh, kw, stride_h, stride_w, pad_h=0, pad_w=0):
    BATCH, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1
    
    y = torch.empty(BATCH, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    TOTAL = BATCH * C * OUT_H * OUT_W
    
    grid = lambda meta: (cdiv(TOTAL, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x, y,
        BATCH, C, IN_H, IN_W,
        OUT_H, OUT_W,
        kh, kw,
        stride_h, stride_w,
        pad_h, pad_w,
    )
    return y


# ============================================================
# Fused Linear + Bias + ReLU kernel (GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'USE_RELU'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    USE_RELU: tl.constexpr,
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
        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        
        # W is (N, K): W[n, k]
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)
        
        acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=False)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=mask_out)


def linear_bias_relu(x, w, b, use_relu=True):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N, meta['BLOCK_N']))
    
    linear_bias_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        use_relu,
    )
    return y


# ============================================================
# Main kernel_function: AlexNet forward pass
# ============================================================
def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_conv4_weight, p_conv4_bias,
    p_conv5_weight, p_conv5_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    # Conv1: in=3, out=96, kernel=11, stride=4, pad=2 -> (1024, 96, 55, 55)
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, 4, 4, 2, 2, use_relu=True)
    
    # MaxPool1: kernel=3, stride=2 -> (1024, 96, 27, 27)
    x = maxpool2d(x, 3, 3, 2, 2, 0, 0)
    
    # Conv2: in=96, out=256, kernel=5, stride=1, pad=2 -> (1024, 256, 27, 27)
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, 1, 1, 2, 2, use_relu=True)
    
    # MaxPool2: kernel=3, stride=2 -> (1024, 256, 13, 13)
    x = maxpool2d(x, 3, 3, 2, 2, 0, 0)
    
    # Conv3: in=256, out=384, kernel=3, stride=1, pad=1 -> (1024, 384, 13, 13)
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, 1, 1, 1, 1, use_relu=True)
    
    # Conv4: in=384, out=384, kernel=3, stride=1, pad=1 -> (1024, 384, 13, 13)
    x = conv2d_bias_relu(x, p_conv4_weight, p_conv4_bias, 1, 1, 1, 1, use_relu=True)
    
    # Conv5: in=384, out=256, kernel=3, stride=1, pad=1 -> (1024, 256, 13, 13)
    x = conv2d_bias_relu(x, p_conv5_weight, p_conv5_bias, 1, 1, 1, 1, use_relu=True)
    
    # MaxPool3: kernel=3, stride=2 -> (1024, 256, 6, 6)
    x = maxpool2d(x, 3, 3, 2, 2, 0, 0)
    
    # Flatten: (1024, 9216)
    x = x.reshape(x.shape[0], -1)
    
    # FC1: (1024, 9216) -> (1024, 4096) + ReLU
    x = linear_bias_relu(x, p_fc1_weight, p_fc1_bias, use_relu=True)
    
    # Dropout (p=0.0, no-op in inference)
    # FC2: (1024, 4096) -> (1024, 4096) + ReLU
    x = linear_bias_relu(x, p_fc2_weight, p_fc2_bias, use_relu=True)
    
    # Dropout (p=0.0, no-op in inference)
    # FC3: (1024, 4096) -> (1024, 1000)
    x = linear_bias_relu(x, p_fc3_weight, p_fc3_bias, use_relu=False)
    
    return x


def get_inputs():
    batch_size = 1024
    p_conv1_weight = torch.randn(96, 3, 11, 11, device='cuda', dtype=torch.float32) * 0.01
    p_conv1_bias = torch.zeros(96, device='cuda', dtype=torch.float32)
    p_conv2_weight = torch.randn(256, 96, 5, 5, device='cuda', dtype=torch.float32) * 0.01
    p_conv2_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_conv3_weight = torch.randn(384, 256, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_conv3_bias = torch.zeros(384, device='cuda', dtype=torch.float32)
    p_conv4_weight = torch.randn(384, 384, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_conv4_bias = torch.zeros(384, device='cuda', dtype=torch.float32)
    p_conv5_weight = torch.randn(256, 384, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_conv5_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_fc1_weight = torch.randn(4096, 9216, device='cuda', dtype=torch.float32) * 0.01
    p_fc1_bias = torch.zeros(4096, device='cuda', dtype=torch.float32)
    p_fc2_weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float32) * 0.01
    p_fc2_bias = torch.zeros(4096, device='cuda', dtype=torch.float32)
    p_fc3_weight = torch.randn(1000, 4096, device='cuda', dtype=torch.float32) * 0.01
    p_fc3_bias = torch.zeros(1000, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, 3, 224, 224, device='cuda', dtype=torch.float32)
    return [
        p_conv1_weight, p_conv1_bias,
        p_conv2_weight, p_conv2_bias,
        p_conv3_weight, p_conv3_bias,
        p_conv4_weight, p_conv4_bias,
        p_conv5_weight, p_conv5_bias,
        p_fc1_weight, p_fc1_bias,
        p_fc2_weight, p_fc2_bias,
        p_fc3_weight, p_fc3_bias,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    if not args.profile_only:
        # Run our kernel
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        
        # Reference using PyTorch
        import torch.nn.functional as F
        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_conv3_weight, p_conv3_bias,
         p_conv4_weight, p_conv4_bias,
         p_conv5_weight, p_conv5_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs
        
        ref = F.conv2d(x, p_conv1_weight, p_conv1_bias, stride=4, padding=2)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)
        ref = F.conv2d(ref, p_conv2_weight, p_conv2_bias, stride=1, padding=2)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)
        ref = F.conv2d(ref, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.conv2d(ref, p_conv4_weight, p_conv4_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.conv2d(ref, p_conv5_weight, p_conv5_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)
        ref = ref.flatten(1)
        ref = F.linear(ref, p_fc1_weight, p_fc1_bias)
        ref = F.relu(ref)
        ref = F.linear(ref, p_fc2_weight, p_fc2_bias)
        ref = F.relu(ref)
        ref = F.linear(ref, p_fc3_weight, p_fc3_bias)
        
        torch.cuda.synchronize()
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Profile-only mode: kernel executed successfully.")