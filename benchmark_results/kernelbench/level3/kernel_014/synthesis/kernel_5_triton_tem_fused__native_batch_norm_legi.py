import torch
import triton
import triton.language as tl
import math

# BN+ReLU fused pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # offs = n * C * HW + c * HW + hw
    # c = (offs // HW) % C
    hw = offs % HW
    tmp = offs // HW
    c = tmp % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * w + b
    x_relu = tl.maximum(x_scaled, 0.0)
    
    tl.store(out_ptr + offs, x_relu, mask=mask)


# Implicit GEMM Conv2d kernel: BN+ReLU already applied to input
# Input: (N, C_in, H, W) in NCHW
# Weight: (C_out, C_in, KH, KW)
# Output: (N, C_out, H_out, W_out) in NCHW
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'C_OUT', 'C_IN'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, C_IN, IN_H, IN_W,
    C_OUT, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M_TOTAL, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp_m = offs_m // OUT_W
    out_h_idx = tmp_m % OUT_H
    batch_idx = tmp_m // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = C_IN * KH * KW
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (c_in, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        cin_idx = tmp_k // KH
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        
        mask_x = valid_m & valid_k & valid_h & valid_w & valid_batch
        
        # x: (BATCH, C_IN, IN_H, IN_W) -> linear index
        x_idx = batch_idx[:, None] * (C_IN * IN_H * IN_W) + cin_idx[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)
        
        # w: (C_OUT, C_IN, KH, KW) -> linear index
        # offs_n: C_OUT indices, offs_k: (cin, kh, kw) indices
        # w[c_out, c_in, kh, kw] = w_ptr[c_out * C_IN*KH*KW + c_in*KH*KW + kh*KW + kw]
        # = w_ptr[offs_n * C_IN*KH*KW + offs_k]  (since offs_k encodes cin*KH*KW + kh*KW + kw)
        valid_n = offs_n[None, :] < C_OUT
        mask_w = valid_k[:, None] & valid_n
        
        w_idx = offs_n[None, :] * TOTAL_K + offs_k[:, None]
        w_tile = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Store output: (BATCH, C_OUT, OUT_H, OUT_W)
    out_idx = batch_idx[:, None] * (C_OUT * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < C_OUT)
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=mask_out)


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N_total = x.numel()
    C = mean.shape[0]
    HW = x.shape[2] * x.shape[3]
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        C, HW,
        N=N_total,
        eps=eps,
    )
    return out


def conv2d_triton(x, w, stride=1, padding=1):
    BATCH, C_IN, IN_H, IN_W = x.shape
    C_OUT, _, KH, KW = w.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    out = torch.empty((BATCH, C_OUT, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C_OUT, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        x, w, out,
        BATCH, C_IN, IN_H, IN_W,
        C_OUT, OUT_H, OUT_W,
        KH, KW,
        stride, stride,
        padding, padding,
        M_TOTAL,
    )
    return out


def kernel_function(
    p_layers_0_0_weight, p_layers_0_0_bias,
    p_layers_0_2_weight,
    p_layers_1_0_weight, p_layers_1_0_bias,
    p_layers_1_2_weight,
    p_layers_2_0_weight, p_layers_2_0_bias,
    p_layers_2_2_weight,
    p_layers_3_0_weight, p_layers_3_0_bias,
    p_layers_3_2_weight,
    p_layers_4_0_weight, p_layers_4_0_bias,
    p_layers_4_2_weight,
    p_layers_5_0_weight, p_layers_5_0_bias,
    p_layers_5_2_weight,
    b_layers_0_0_running_mean, b_layers_0_0_running_var,
    b_layers_1_0_running_mean, b_layers_1_0_running_var,
    b_layers_2_0_running_mean, b_layers_2_0_running_var,
    b_layers_3_0_running_mean, b_layers_3_0_running_var,
    b_layers_4_0_running_mean, b_layers_4_0_running_var,
    b_layers_5_0_running_mean, b_layers_5_0_running_var,
    x,
):
    features = [x]
    
    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3) + cat -> [10, 64, 224, 224]
    bn0 = bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
    conv0 = conv2d_triton(bn0, p_layers_0_2_weight)
    features.append(conv0)
    x = torch.cat(features, dim=1)
    
    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3) + cat -> [10, 96, 224, 224]
    bn1 = bn_relu(x, b_layers_1_0_running_mean, b_layers_1_0_running_var,
    conv1 = conv2d_triton(bn1, p_layers_1_2_weight)
    features.append(conv1)
    x = torch.cat(features, dim=1)
    
    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3) + cat -> [10, 128, 224, 224]
    bn2 = bn_relu(x, b_layers_2_0_running_mean, b_layers_2_0_running_var,
    conv2 = conv2d_triton(bn2, p_layers_2_2_weight)
    features.append(conv2)
    x = torch.cat(features, dim=1)
    
    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3) + cat -> [10, 160, 224, 224]
    bn3 = bn_relu(x, b_layers_3_0_running_mean, b_layers_3_0_running_var,
    conv3 = conv2d_triton(bn3, p_layers_3_2_weight)
    features.append(conv3)
    x = torch.cat(features, dim=1)
    
    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3) + cat -> [10, 192, 224, 224]
    bn4 = bn_relu(x, b_layers_4_0_running_mean, b_layers_4_0_running_var,
    conv4 = conv2d_triton(bn4, p_layers_4_2_weight)
    features.append(conv4)
    x = torch.cat(features, dim=1)
    
    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3) + cat -> [10, 224, 224, 224]
    bn5 = bn_relu(x, b_layers_5_0_running_mean, b_layers_5_0_running_var,
    conv5 = conv2d_triton(bn5, p_layers_5_2_weight)
    features.append(conv5)
    x = torch.cat(features, dim=1)
    
    return x


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    num_layers = 6
    
    p_layers_0_0_weight = torch.ones(32, device='cuda')
    p_layers_0_0_bias = torch.zeros(32, device='cuda')
    p_layers_0_2_weight = torch.randn(32, 32, 3, 3, device='cuda') * 0.1
    
    p_layers_1_0_weight = torch.ones(64, device='cuda')
    p_layers_1_0_bias = torch.zeros(64, device='cuda')
    p_layers_1_2_weight = torch.randn(32, 64, 3, 3, device='cuda') * 0.1
    
    p_layers_2_0_weight = torch.ones(96, device='cuda')
    p_layers_2_0_bias = torch.zeros(96, device='cuda')
    p_layers_2_2_weight = torch.randn(32, 96, 3, 3, device='cuda') * 0.1
    
    p_layers_3_0_weight = torch.ones(128, device='cuda')
    p_layers_3_0_bias = torch.zeros(128, device='cuda')
    p_layers_3_2_weight = torch.randn(32, 128, 3, 3, device='cuda') * 0.1
    
    p_layers_4_0_weight = torch.ones(160, device='cuda')
    p_layers_4_0_bias = torch.zeros(160, device='cuda')
    p_layers_4_2_weight = torch.randn(32, 160, 3, 3, device='cuda') * 0.1
    
    p_layers_5_0_weight = torch.ones(192, device='cuda')
    p_layers_5_0_bias = torch.zeros(192, device='cuda')
    p_layers_5_2_weight = torch.randn(32, 192, 3, 3, device='cuda') * 0.1
    
    b_layers_0_0_running_mean = torch.zeros(32, device='cuda')
    b_layers_0_0_running_var = torch.ones(32, device='cuda')
    b_layers_1_0_running_mean = torch.zeros(64, device='cuda')
    b_layers_1_0_running_var = torch.ones(64, device='cuda')
    b_layers_2_0_running_mean = torch.zeros(96, device='cuda')
    b_layers_2_0_running_var = torch.ones(96, device='cuda')
    b_layers_3_0_running_mean = torch.zeros(128, device='cuda')
    b_layers_3_0_running_var = torch.ones(128, device='cuda')
    b_layers_4_0_running_mean = torch.zeros(160, device='cuda')
    b_layers_4_0_running_var = torch.ones(160, device='cuda')
    b_layers_5_0_running_mean = torch.zeros(192, device='cuda')
    b_layers_5_0_running_var = torch.ones(192, device='cuda')
    
    x = torch.rand(batch_size, num_input_features, height, width, device='cuda')
    
    return [
        p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
        p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
        p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
        p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
        p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
        p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
        b_layers_0_0_running_mean, b_layers_0_0_running_var,
        b_layers_1_0_running_mean, b_layers_1_0_running_var,
        b_layers_2_0_running_mean, b_layers_2_0_running_var,
        b_layers_3_0_running_mean, b_layers_3_0_running_var,
        b_layers_4_0_running_mean, b_layers_4_0_running_var,
        b_layers_5_0_running_mean, b_layers_5_0_running_var,
        x,
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
        import torch.nn.functional as F
        
        # Reference implementation
        (p_layers_0_0_weight, p_layers_0_0_bias, p_layers_0_2_weight,
         p_layers_1_0_weight, p_layers_1_0_bias, p_layers_1_2_weight,
         p_layers_2_0_weight, p_layers_2_0_bias, p_layers_2_2_weight,
         p_layers_3_0_weight, p_layers_3_0_bias, p_layers_3_2_weight,
         p_layers_4_0_weight, p_layers_4_0_bias, p_layers_4_2_weight,
         p_layers_5_0_weight, p_layers_5_0_bias, p_layers_5_2_weight,
         b_layers_0_0_running_mean, b_layers_0_0_running_var,
         b_layers_1_0_running_mean, b_layers_1_0_running_var,
         b_layers_2_0_running_mean, b_layers_2_0_running_var,
         b_layers_3_0_running_mean, b_layers_3_0_running_var,
         b_layers_4_0_running_mean, b_layers_4_0_running_var,
         b_layers_5_0_running_mean, b_layers_5_0_running_var,
         x) = inputs
        
        def ref_bn_relu(inp, rm, rv, w, b):
            return F.relu(F.batch_norm(inp, rm, rv, w, b, training=False, eps=1e-5))
        
        features = [x]
        bn0 = ref_bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
        conv0 = F.conv2d(bn0, p_layers_0_2_weight, padding=1)
        features.append(conv0)
        xr = torch.cat(features, dim=1)
        
        bn1 = ref_bn_relu(xr, b_layers_1_0_running_mean, b_layers_1_0_running_var,
        conv1 = F.conv2d(bn1, p_layers_1_2_weight, padding=1)
        features.append(conv1)
        xr = torch.cat(features, dim=1)
        
        bn2 = ref_bn_relu(xr, b_layers_2_0_running_mean, b_layers_2_0_running_var,
        conv2 = F.conv2d(bn2, p_layers_2_2_weight, padding=1)
        features.append(conv2)
        xr = torch.cat(features, dim=1)
        
        bn3 = ref_bn_relu(xr, b_layers_3_0_running_mean, b_layers_3_0_running_var,
        conv3 = F.conv2d(bn3, p_layers_3_2_weight, padding=1)
        features.append(conv3)
        xr = torch.cat(features, dim=1)
        
        bn4 = ref_bn_relu(xr, b_layers_4_0_running_mean, b_layers_4_0_running_var,
        conv4 = F.conv2d(bn4, p_layers_4_2_weight, padding=1)
        features.append(conv4)
        xr = torch.cat(features, dim=1)
        
        bn5 = ref_bn_relu(xr, b_layers_5_0_running_mean, b_layers_5_0_running_var,
        conv5 = F.conv2d(bn5, p_layers_5_2_weight, padding=1)
        features.append(conv5)
        ref = torch.cat(features, dim=1)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")