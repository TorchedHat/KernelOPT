import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    apply_relu: tl.constexpr,
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
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    if apply_relu:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'OUT_C', 'OUT_H', 'OUT_W', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    M_total = BATCH * OUT_H * OUT_W
    
    # Decode M indices
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K: k = ic * KH * KW + kh * KW + kw
        kw_idx = k_offs % KERNEL_W
        tmp_k = k_offs // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        ic_idx = tmp_k // KERNEL_H
        
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid = (
            (offs_m[:, None] < M_total) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (k_offs[None, :] < TOTAL_K)
        )
        
        # X in NCHW: X[n, ic, ih, iw] = X_ptr + n*IN_C*IN_H*IN_W + ic*IN_H*IN_W + ih*IN_W + iw
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * (IN_C * IN_H * IN_W) +
                  ic_idx[None, :] * (IN_H * IN_W) +
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # W in NCHW: W[oc, ic, kh, kw] = W_ptr + oc*IN_C*KH*KW + ic*KH*KW + kh*KW + kw
        w_ptrs = (W_ptr +
                  offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
                  ic_idx[:, None] * (KERNEL_H * KERNEL_W) +
                  kh_idx[:, None] * KERNEL_W +
                  kw_idx[:, None])
        w_mask = (k_offs[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    if apply_relu:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
    
    # Store in NCHW: Out[n, oc, oh, ow]
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
                offs_n[None, :] * (OUT_H * OUT_W) +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Out_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W, STRIDE_H, STRIDE_W,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode: offs = n * C * OUT_H * OUT_W + c * OUT_H * OUT_W + oh * OUT_W + ow
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_ptrs = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x_val)
    
    out_ptrs = Out_ptr + offs
    tl.store(out_ptrs, max_val, mask=mask)


def conv2d_relu(x, weight, bias, stride=1, padding=1, apply_relu=True):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    
    out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M = N * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_relu_kernel[grid](
        x, weight, bias, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        KH, KW,
        apply_relu,
    )
    return out


def maxpool2d(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H - kernel_size) // stride + 1
    OUT_W = (IN_W - kernel_size) // stride + 1
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    N_total = N * C * OUT_H * OUT_W
    
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride,
        N_total,
    )
    return out


def linear_relu(x, weight, bias, apply_relu=True):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        apply_relu,
    )
    return out


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_16_weight, p_features_16_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_23_weight, p_features_23_bias,
    p_features_25_weight, p_features_25_bias,
    p_features_28_weight, p_features_28_bias,
    p_features_30_weight, p_features_30_bias,
    p_features_32_weight, p_features_32_bias,
    p_features_34_weight, p_features_34_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    # Block 1
    x = conv2d_relu(x, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 2
    x = conv2d_relu(x, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 3
    x = conv2d_relu(x, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_16_weight, p_features_16_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 4
    x = conv2d_relu(x, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_23_weight, p_features_23_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_25_weight, p_features_25_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 5
    x = conv2d_relu(x, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_30_weight, p_features_30_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_32_weight, p_features_32_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_34_weight, p_features_34_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Flatten
    x = x.reshape(x.shape[0], -1)
    
    # Classifier
    x = linear_relu(x, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    x = linear_relu(x, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    x = linear_relu(x, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)
    
    return x


def get_inputs():
    torch.manual_seed(42)
    return [
        torch.randn(64, 3, 3, 3, device='cuda'),
        torch.randn(64, device='cuda'),
        torch.randn(64, 64, 3, 3, device='cuda'),
        torch.randn(64, device='cuda'),
        torch.randn(128, 64, 3, 3, device='cuda'),
        torch.randn(128, device='cuda'),
        torch.randn(128, 128, 3, 3, device='cuda'),
        torch.randn(128, device='cuda'),
        torch.randn(256, 128, 3, 3, device='cuda'),
        torch.randn(256, device='cuda'),
        torch.randn(256, 256, 3, 3, device='cuda'),
        torch.randn(256, device='cuda'),
        torch.randn(256, 256, 3, 3, device='cuda'),
        torch.randn(256, device='cuda'),
        torch.randn(256, 256, 3, 3, device='cuda'),
        torch.randn(256, device='cuda'),
        torch.randn(512, 256, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(512, 512, 3, 3, device='cuda'),
        torch.randn(512, device='cuda'),
        torch.randn(4096, 25088, device='cuda'),
        torch.randn(4096, device='cuda'),
        torch.randn(4096, 4096, device='cuda'),
        torch.randn(4096, device='cuda'),
        torch.randn(1000, 4096, device='cuda'),
        torch.randn(1000, device='cuda'),
        torch.randn(10, 3, 224, 224, device='cuda'),
    ]


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
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Set weights
        conv_layers = [l for l in model if isinstance(l, nn.Conv2d)]
        weights = [
            (inputs[0], inputs[1]),
            (inputs[2], inputs[3]),
            (inputs[4], inputs[5]),
            (inputs[6], inputs[7]),
            (inputs[8], inputs[9]),
            (inputs[10], inputs[11]),
            (inputs[12], inputs[13]),
            (inputs[14], inputs[15]),
            (inputs[16], inputs[17]),
            (inputs[18], inputs[19]),
            (inputs[20], inputs[21]),
            (inputs[22], inputs[23]),
            (inputs[24], inputs[25]),
            (inputs[26], inputs[27]),
            (inputs[28], inputs[29]),
            (inputs[30], inputs[31]),
        ]
        for layer, (w, b) in zip(conv_layers, weights):
            layer.weight.data = w.clone()
            layer.bias.data = b.clone()
        
        model = model.cuda()
        with torch.no_grad():
            feat = model(inputs[-1])
            feat = feat.flatten(1)
            
            # FC layers
            ref = feat @ inputs[32].T + inputs[33]
            ref = torch.relu(ref)
            ref = ref @ inputs[34].T + inputs[35]
            ref = torch.relu(ref)
            ref = ref @ inputs[36].T + inputs[37]
        
        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")