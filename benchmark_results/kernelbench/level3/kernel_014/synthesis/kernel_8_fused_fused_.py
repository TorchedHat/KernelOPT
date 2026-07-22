import torch
import triton
import triton.language as tl
import math

# BatchNorm + ReLU fused kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_ELEMENTS', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    # Total elements = N * C * HW
    # For each element, channel = (flat_idx // HW) % C
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N_ELEMENTS = N * C * HW
    mask = offsets < N_ELEMENTS
    
    # Compute channel index
    hw_idx = offsets % HW
    nc_idx = offsets // HW
    c_idx = nc_idx % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Implicit GEMM Conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # Strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) for coalescing
    s_wo, s_wkh, s_wkw, s_wi,
    # Strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K index -> (kh, kw, in_c) - stored as (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        
        x_mask = valid_m & valid_k & valid_spatial & valid_batch
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * s_xn + 
                  in_c[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Weight: stored as (OUT_C, KH, KW, IN_C) for coalescing
        valid_n = offs_n[None, :] < OUT_C
        valid_k2 = offs_k[:, None] < TOTAL_K
        w_mask = valid_k2 & valid_n
        
        w_ptrs = (W_ptr + 
                  offs_n[None, :] * s_wo + 
                  kh[:, None] * s_wkh + 
                  kw[:, None] * s_wkw + 
                  in_c[:, None] * s_wi)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output
    valid_m_out = offs_m[:, None] < M_TOTAL
    valid_n_out = offs_n[None, :] < OUT_C
    out_mask = valid_m_out & valid_n_out
    
    y_ptrs = (Y_ptr + 
              batch_idx[:, None] * s_yn + 
              out_h_idx[:, None] * s_yh + 
              out_w_idx[:, None] * s_yw + 
              offs_n[None, :] * s_yc)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    N_ELEMENTS = N * C * H * W
    HW = H * W
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        N, C, HW,
        eps=eps,
        N_ELEMENTS=N_ELEMENTS,
    )
    return out


def conv2d_triton(x_nhwc, w, OUT_H, OUT_W, STRIDE_H=1, STRIDE_W=1, PAD_H=1, PAD_W=1):
    """
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape
    
    # Reorder weight to (OUT_C, KH, KW, IN_C) for coalesced access
    w_reordered = w.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)
    
    # Output in NHWC layout
    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    
    M_TOTAL = N * OUT_H * OUT_W
    
    # Strides for x_nhwc (N, H, W, C)
    s_xn = x_nhwc.stride(0)
    s_xh = x_nhwc.stride(1)
    s_xw = x_nhwc.stride(2)
    s_xc = x_nhwc.stride(3)
    
    # Strides for w_reordered (OUT_C, KH, KW, IN_C)
    s_wo = w_reordered.stride(0)
    s_wkh = w_reordered.stride(1)
    s_wkw = w_reordered.stride(2)
    s_wi = w_reordered.stride(3)
    
    # Strides for y (N, H, W, C)
    s_yn = y.stride(0)
    s_yh = y.stride(1)
    s_yw = y.stride(2)
    s_yc = y.stride(3)
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_reordered, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wi,
        s_yn, s_yh, s_yw, s_yc,
        BATCH=N, IN_C=IN_C, OUT_C=OUT_C, OUT_H=OUT_H, OUT_W=OUT_W,
    )
    
    return y


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
    N, C, H, W = x.shape  # (10, 32, 224, 224)
    
    # Layer 0: BN(32) + ReLU + Conv(32->32, 3x3) + Cat -> (10, 64, 224, 224)
    # BN + ReLU on x (10, 32, 224, 224)
    bn0_out = bn_relu(x, b_layers_0_0_running_mean, b_layers_0_0_running_var,
    # Convert to NHWC for conv
    bn0_nhwc = bn0_out.permute(0, 2, 3, 1).contiguous()
    conv0_nhwc = conv2d_triton(bn0_nhwc, p_layers_0_2_weight, H, W)
    conv0 = conv0_nhwc.permute(0, 3, 1, 2).contiguous()
    # Cat: [x, conv0] -> (10, 64, 224, 224)
    x1 = torch.cat([x, conv0], dim=1)
    
    # Layer 1: BN(64) + ReLU + Conv(64->32, 3x3) + Cat -> (10, 96, 224, 224)
    bn1_out = bn_relu(x1, b_layers_1_0_running_mean, b_layers_1_0_running_var,
    bn1_nhwc = bn1_out.permute(0, 2, 3, 1).contiguous()
    conv1_nhwc = conv2d_triton(bn1_nhwc, p_layers_1_2_weight, H, W)
    conv1 = conv1_nhwc.permute(0, 3, 1, 2).contiguous()
    x2 = torch.cat([x1, conv1], dim=1)
    
    # Layer 2: BN(96) + ReLU + Conv(96->32, 3x3) + Cat -> (10, 128, 224, 224)
    bn2_out = bn_relu(x2, b_layers_2_0_running_mean, b_layers_2_0_running_var,
    bn2_nhwc = bn2_out.permute(0, 2, 3, 1).contiguous()
    conv2_nhwc = conv2d_triton(bn2_nhwc, p_layers_2_2_weight, H, W)
    conv2 = conv2_nhwc.permute(0, 3, 1, 2).contiguous()
    x3 = torch.cat([x2, conv2], dim=1)
    
    # Layer 3: BN(128) + ReLU + Conv(128->32, 3x3) + Cat -> (10, 160, 224, 224)
    bn3_out = bn_relu(x3, b_layers_3_0_running_mean, b_layers_3_0_running_var,
    bn3_nhwc = bn3_out.permute(0, 2, 3, 1).contiguous()
    conv3_nhwc = conv2d_triton(bn3_nhwc, p_layers_3_2_weight, H, W)
    conv3 = conv3_nhwc.permute(0, 3, 1, 2).contiguous()
    x4 = torch.cat([x3, conv3], dim=1)
    
    # Layer 4: BN(160) + ReLU + Conv(160->32, 3x3) + Cat -> (10, 192, 224, 224)
    bn4_out = bn_relu(x4, b_layers_4_0_running_mean, b_layers_4_0_running_var,
    bn4_nhwc = bn4_out.permute(0, 2, 3, 1).contiguous()
    conv4_nhwc = conv2d_triton(bn4_nhwc, p_layers_4_2_weight, H, W)
    conv4 = conv4_nhwc.permute(0, 3, 1, 2).contiguous()
    x5 = torch.cat([x4, conv4], dim=1)
    
    # Layer 5: BN(192) + ReLU + Conv(192->32, 3x3) + Cat -> (10, 224, 224, 224)
    bn5_out = bn_relu(x5, b_layers_5_0_running_mean, b_layers_5_0_running_var,
    bn5_nhwc = bn5_out.permute(0, 2, 3, 1).contiguous()
    conv5_nhwc = conv2d_triton(bn5_nhwc, p_layers_5_2_weight, H, W)
    conv5 = conv5_nhwc.permute(0, 3, 1, 2).contiguous()
    x6 = torch.cat([x5, conv5], dim=1)
    
    return x6


def get_inputs():
    batch_size = 10
    num_input_features = 32
    growth_rate = 32
    height, width = 224, 224
    num_layers = 6
    
    device = 'cuda'
    
    p_layers_0_0_weight = torch.ones(32, device=device)
    p_layers_0_0_bias = torch.zeros(32, device=device)
    p_layers_0_2_weight = torch.randn(32, 32, 3, 3, device=device) * 0.1
    
    p_layers_1_0_weight = torch.ones(64, device=device)
    p_layers_1_0_bias = torch.zeros(64, device=device)
    p_layers_1_2_weight = torch.randn(32, 64, 3, 3, device=device) * 0.1
    
    p_layers_2_0_weight = torch.ones(96, device=device)
    p_layers_2_0_bias = torch.zeros(96, device=device)
    p_layers_2_2_weight = torch.randn(32, 96, 3, 3, device=device) * 0.1
    
    p_layers_3_0_weight = torch.ones(128, device=device)
    p_layers_3_0_bias = torch.zeros(128, device=device)
    p_layers_3_2_weight = torch.randn(32, 128, 3, 3, device=device) * 0.1
    
    p_layers_4_0_weight = torch.ones(160, device=device)
    p_layers_4_0_bias = torch.zeros(160, device=device)
    p_layers_4_2_weight = torch.randn(32, 160, 3, 3, device=device) * 0.1
    
    p_layers_5_0_weight = torch.ones(192, device=device)
    p_layers_5_0_bias = torch.zeros(192, device=device)
    p_layers_5_2_weight = torch.randn(32, 192, 3, 3, device=device) * 0.1
    
    b_layers_0_0_running_mean = torch.zeros(32, device=device)
    b_layers_0_0_running_var = torch.ones(32, device=device)
    b_layers_1_0_running_mean = torch.zeros(64, device=device)
    b_layers_1_0_running_var = torch.ones(64, device=device)
    b_layers_2_0_running_mean = torch.zeros(96, device=device)
    b_layers_2_0_running_var = torch.ones(96, device=device)
    b_layers_3_0_running_mean = torch.zeros(128, device=device)
    b_layers_3_0_running_var = torch.ones(128, device=device)
    b_layers_4_0_running_mean = torch.zeros(160, device=device)
    b_layers_4_0_running_var = torch.ones(160, device=device)
    b_layers_5_0_running_mean = torch.zeros(192, device=device)
    b_layers_5_0_running_var = torch.ones(192, device=device)
    
    x = torch.rand(batch_size, num_input_features, height, width, device=device)
    
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
        
        # Build reference model
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
        
        def ref_bn_relu(inp, mean, var, weight, bias, eps=1e-5):
            x_norm = (inp - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps)
            return torch.relu(x_norm * weight[None, :, None, None] + bias[None, :, None, None])
        
        def ref_conv2d(inp, weight):
            # Manual conv2d using unfold
            N, C, H, W = inp.shape
            OUT_C, IN_C, KH, KW = weight.shape
            # Use F.conv2d - but that's banned... use manual approach
            # Actually for reference we can use it
            import torch.nn.functional as F
            return F.conv2d(inp, weight, padding=1)
        
        with torch.no_grad():
            x0 = x
            bn0 = ref_bn_relu(x0, b_layers_0_0_running_mean, b_layers_0_0_running_var,
            c0 = ref_conv2d(bn0, p_layers_0_2_weight)
            x1 = torch.cat([x0, c0], dim=1)
            
            bn1 = ref_bn_relu(x1, b_layers_1_0_running_mean, b_layers_1_0_running_var,
            c1 = ref_conv2d(bn1, p_layers_1_2_weight)
            x2 = torch.cat([x1, c1], dim=1)
            
            bn2 = ref_bn_relu(x2, b_layers_2_0_running_mean, b_layers_2_0_running_var,
            c2 = ref_conv2d(bn2, p_layers_2_2_weight)
            x3 = torch.cat([x2, c2], dim=1)
            
            bn3 = ref_bn_relu(x3, b_layers_3_0_running_mean, b_layers_3_0_running_var,
            c3 = ref_conv2d(bn3, p_layers_3_2_weight)
            x4 = torch.cat([x3, c3], dim=1)
            
            bn4 = ref_bn_relu(x4, b_layers_4_0_running_mean, b_layers_4_0_running_var,
            c4 = ref_conv2d(bn4, p_layers_4_2_weight)
            x5 = torch.cat([x4, c4], dim=1)
            
            bn5 = ref_bn_relu(x5, b_layers_5_0_running_mean, b_layers_5_0_running_var,
            c5 = ref_conv2d(bn5, p_layers_5_2_weight)
            ref = torch.cat([x5, c5], dim=1)
        
        assert torch.allclose(output, ref, rtol=1e-2, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")