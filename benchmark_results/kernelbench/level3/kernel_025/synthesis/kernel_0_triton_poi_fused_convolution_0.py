import torch
import triton
import triton.language as tl
import math

# BN + ReLU fused kernel
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
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # channel index
    c = (offsets // HW) % C
    
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
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Channel shuffle kernel: input (N, C, H, W) -> output (N, C, H, W)
# Shuffle: view as (N, groups, C//groups, H, W), transpose(1,2) -> (N, C//groups, groups, H, W), view back
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
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # output index: (n, c_out, h, w) in NCHW
    HW = H * W
    CHW = C * HW
    
    n = offsets // CHW
    rem = offsets % CHW
    c_out = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W
    
    # channel shuffle: output channel c_out came from input channel c_in
    # c_out = (c_in % groups) * (C // groups) + c_in // groups
    # inverse: c_in = (c_out % (C // groups)) * groups + c_out // (C // groups)
    cpg = C // groups  # channels per group
    c_in = (c_out % cpg) * groups + c_out // cpg
    
    in_idx = n * CHW + c_in * HW + h * W + w
    
    val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


# BN + add (residual) fused kernel
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
def bn_add_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Grouped conv2d 1x1 kernel (implicit GEMM)
# Input: (N, C_in, H, W), Weight: (C_out, C_in//groups, 1, 1)
# Output: (N, C_out, H, W)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def grouped_conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, C_out, HW,
    groups,
    M, K, # M = N*HW, K = C_in//groups (per group)
    stride_xn, stride_xc, stride_xhw,
    stride_wco, stride_wci,
    stride_on, stride_oc, stride_ohw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0) = tile over M (N*HW)
    # program_id(1) = tile over N (C_out per group)
    # program_id(2) = group index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)
    
    GROUP_IN_C = C_in // groups
    GROUP_OUT_C = C_out // groups
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, hw)
    batch = offs_m // HW
    hw = offs_m % HW
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, GROUP_IN_C, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Load input: x[batch, pid_g*GROUP_IN_C + offs_k, hw]
        c_in = pid_g * GROUP_IN_C + offs_k
        x_ptrs = x_ptr + batch[:, None] * stride_xn + c_in[None, :] * stride_xc + hw[:, None] * stride_xhw
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < GROUP_IN_C)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Load weight: w[pid_g*GROUP_OUT_C + offs_n, offs_k, 0, 0]
        c_out = pid_g * GROUP_OUT_C + offs_n
        w_ptrs = w_ptr + c_out[None, :] * stride_wco + offs_k[:, None] * stride_wci
        mask_w = (offs_k[:, None] < GROUP_IN_C) & (offs_n[None, :] < GROUP_OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    c_out_global = pid_g * GROUP_OUT_C + offs_n
    out_ptrs = out_ptr + batch[:, None] * stride_on + c_out_global[None, :] * stride_oc + hw[:, None] * stride_ohw
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < GROUP_OUT_C)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# Depthwise conv2d 3x3 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 1}, num_warps=4),
    ],
    key=['M', 'C'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H, W,
    pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    batch = offs_m // (H * W)
    hw = offs_m % (H * W)
    out_h = hw // W
    out_w = hw % W
    
    c = pid_c
    
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = out_h + kh - pad_h
            iw = out_w + kw - pad_w
            valid = mask_m & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            
            x_ptrs = x_ptr + batch * stride_xn + c * stride_xc + ih * stride_xh + iw * stride_xw
            x_val = tl.load(x_ptrs, mask=valid, other=0.0)
            
            w_val = tl.load(w_ptr + c * stride_wc + kh * stride_wkh + kw * stride_wkw)
            
            acc += x_val * w_val
    
    out_ptrs = out_ptr + batch * stride_on + c * stride_oc + out_h * stride_oh + out_w * stride_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_m)


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
    p_conv3_weight, p_bn3_weight, p_bn3_bias,
    p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    b_bn3_running_mean, b_bn3_running_var,
    b_shortcut_1_running_mean, b_shortcut_1_running_var,
):
    N, C_in, H, W = x.shape  # 10, 240, 224, 224
    groups = 3
    mid_channels = 120  # C_out of conv1
    out_channels = 480
    eps = 1e-5
    
    # Step 1: Grouped conv1x1 (groups=3): x(10,240,224,224) -> (10,120,224,224)
    HW = H * W
    M = N * HW
    GROUP_IN_C1 = C_in // groups  # 80
    GROUP_OUT_C1 = mid_channels // groups  # 40
    
    conv1_out = torch.empty(N, mid_channels, H, W, device=x.device, dtype=x.dtype)
    
    grid1 = (
        triton.cdiv(M, 128),
        triton.cdiv(GROUP_OUT_C1, 64),
        groups,
    )
    grouped_conv1x1_kernel[grid1](
        x, p_conv1_weight, conv1_out,
        N, C_in, mid_channels, HW,
        groups,
        M, GROUP_IN_C1,
        x.stride(0), x.stride(1), x.stride(2) * W + x.stride(3),  # stride_xhw = stride per hw element
        p_conv1_weight.stride(0), p_conv1_weight.stride(1),
        conv1_out.stride(0), conv1_out.stride(1), conv1_out.stride(2) * W + conv1_out.stride(3),
    )
    
    # Step 2: BN + ReLU
    bn1_out = torch.empty_like(conv1_out)
    numel1 = conv1_out.numel()
    grid_bn1 = (triton.cdiv(numel1, 1024),)
    bn_relu_kernel[grid_bn1](
        conv1_out, bn1_out,
        b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias,
        numel1, mid_channels, HW,
        eps,
    )
    
    # Step 3: Depthwise conv2d 3x3 (groups=mid_channels=120)
    conv2_out = torch.empty(N, mid_channels, H, W, device=x.device, dtype=x.dtype)
    
    grid_dw = (triton.cdiv(M, 256), mid_channels)
    depthwise_conv3x3_kernel[grid_dw](
        bn1_out, p_conv2_weight, conv2_out,
        N, mid_channels, H, W,
        M,
        bn1_out.stride(0), bn1_out.stride(1), bn1_out.stride(2), bn1_out.stride(3),
        p_conv2_weight.stride(0), p_conv2_weight.stride(2), p_conv2_weight.stride(3),
        conv2_out.stride(0), conv2_out.stride(1), conv2_out.stride(2), conv2_out.stride(3),
    )
    
    # Step 4: BN (no relu)
    bn2_out = torch.empty_like(conv2_out)
    numel2 = conv2_out.numel()
    grid_bn2 = (triton.cdiv(numel2, 1024),)
    bn_kernel[grid_bn2](
        conv2_out, bn2_out,
        b_bn2_running_mean, b_bn2_running_var, p_bn2_weight, p_bn2_bias,
        numel2, mid_channels, HW,
        eps,
    )
    
    # Step 5: Channel shuffle (groups=3)
    shuffled = torch.empty_like(bn2_out)
    numel_s = bn2_out.numel()
    grid_s = (triton.cdiv(numel_s, 1024),)
    channel_shuffle_kernel[grid_s](
        bn2_out, shuffled,
        N, mid_channels, H, W,
        groups,
        numel_s,
    )
    
    # Step 6: Grouped conv1x1 (groups=3): (10,120,224,224) -> (10,480,224,224)
    GROUP_IN_C3 = mid_channels // groups  # 40
    GROUP_OUT_C3 = out_channels // groups  # 160
    
    conv3_out = torch.empty(N, out_channels, H, W, device=x.device, dtype=x.dtype)
    
    grid3 = (
        triton.cdiv(M, 128),
        triton.cdiv(GROUP_OUT_C3, 64),
        groups,
    )
    grouped_conv1x1_kernel[grid3](
        shuffled, p_conv3_weight, conv3_out,
        N, mid_channels, out_channels, HW,
        groups,
        M, GROUP_IN_C3,
        shuffled.stride(0), shuffled.stride(1), shuffled.stride(2) * W + shuffled.stride(3),
        p_conv3_weight.stride(0), p_conv3_weight.stride(1),
        conv3_out.stride(0), conv3_out.stride(1), conv3_out.stride(2) * W + conv3_out.stride(3),
    )
    
    # Step 7: BN + ReLU
    bn3_out = torch.empty_like(conv3_out)
    numel3 = conv3_out.numel()
    grid_bn3 = (triton.cdiv(numel3, 1024),)
    bn_relu_kernel[grid_bn3](
        conv3_out, bn3_out,
        b_bn3_running_mean, b_bn3_running_var, p_bn3_weight, p_bn3_bias,
        numel3, out_channels, HW,
        eps,
    )
    
    # Step 8: Shortcut conv1x1 (no groups): x(10,240,224,224) -> (10,480,224,224)
    GROUP_IN_SC = C_in  # 240
    GROUP_OUT_SC = out_channels  # 480
    
    sc_out = torch.empty(N, out_channels, H, W, device=x.device, dtype=x.dtype)
    
    # Single group conv1x1
    grid_sc = (
        triton.cdiv(M, 128),
        triton.cdiv(GROUP_OUT_SC, 64),
        1,
    )
    grouped_conv1x1_kernel[grid_sc](
        x, p_shortcut_0_weight, sc_out,
        N, C_in, out_channels, HW,
        1,  # groups=1
        M, C_in,
        x.stride(0), x.stride(1), x.stride(2) * W + x.stride(3),
        p_shortcut_0_weight.stride(0), p_shortcut_0_weight.stride(1),
        sc_out.stride(0), sc_out.stride(1), sc_out.stride(2) * W + sc_out.stride(3),
    )
    
    # Step 9: BN + add (residual)
    out = torch.empty_like(bn3_out)
    numel_sc = sc_out.numel()
    grid_sc_bn = (triton.cdiv(numel_sc, 1024),)
    bn_add_kernel[grid_sc_bn](
        sc_out, bn3_out, out,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
        p_shortcut_1_weight, p_shortcut_1_bias,
        numel_sc, out_channels, HW,
        eps,
    )
    
    return out


def get_inputs():
    p_conv1_weight = torch.randn(120, 80, 1, 1, device='cuda')
    p_bn1_weight = torch.ones(120, device='cuda')
    p_bn1_bias = torch.zeros(120, device='cuda')
    p_conv2_weight = torch.randn(120, 1, 3, 3, device='cuda')
    p_bn2_weight = torch.ones(120, device='cuda')
    p_bn2_bias = torch.zeros(120, device='cuda')
    p_conv3_weight = torch.randn(480, 40, 1, 1, device='cuda')
    p_bn3_weight = torch.ones(480, device='cuda')
    p_bn3_bias = torch.zeros(480, device='cuda')
    p_shortcut_0_weight = torch.randn(480, 240, 1, 1, device='cuda')
    p_shortcut_1_weight = torch.ones(480, device='cuda')
    p_shortcut_1_bias = torch.zeros(480, device='cuda')
    b_bn1_running_mean = torch.zeros(120, device='cuda')
    b_bn1_running_var = torch.ones(120, device='cuda')
    b_bn2_running_mean = torch.zeros(120, device='cuda')
    b_bn2_running_var = torch.ones(120, device='cuda')
    b_bn3_running_mean = torch.zeros(480, device='cuda')
    b_bn3_running_var = torch.ones(480, device='cuda')
    b_shortcut_1_running_mean = torch.zeros(480, device='cuda')
    b_shortcut_1_running_var = torch.ones(480, device='cuda')
    x = torch.randn(10, 240, 224, 224, device='cuda')
    return [
        p_conv1_weight, p_bn1_weight, p_bn1_bias,
        p_conv2_weight, p_bn2_weight, p_bn2_bias,
        p_conv3_weight, p_bn3_weight, p_bn3_bias,
        p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
        b_bn1_running_mean, b_bn1_running_var,
        b_bn2_running_mean, b_bn2_running_var,
        b_bn3_running_mean, b_bn3_running_var,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
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
        # Reference implementation
        import torch.nn as nn
        import torch.nn.functional as F
        
        (p_conv1_weight, p_bn1_weight, p_bn1_bias,
         p_conv2_weight, p_bn2_weight, p_bn2_bias,
         p_conv3_weight, p_bn3_weight, p_bn3_bias,
         p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
         b_bn1_running_mean, b_bn1_running_var,
         b_bn2_running_mean, b_bn2_running_var,
         b_bn3_running_mean, b_bn3_running_var,
         b_shortcut_1_running_mean, b_shortcut_1_running_var,
         x) = inputs
        
        # conv1: groups=3
        ref_conv1 = F.conv2d(x, p_conv1_weight, bias=None, stride=1, padding=0, groups=3)
        ref_bn1 = F.batch_norm(ref_conv1, b_bn1_running_mean, b_bn1_running_var,
                               p_bn1_weight, p_bn1_bias, training=False, eps=1e-5)
        ref_relu1 = F.relu(ref_bn1)
        
        # conv2: depthwise
        ref_conv2 = F.conv2d(ref_relu1, p_conv2_weight, bias=None, stride=1, padding=1, groups=120)
        ref_bn2 = F.batch_norm(ref_conv2, b_bn2_running_mean, b_bn2_running_var,
                               p_bn2_weight, p_bn2_bias, training=False, eps=1e-5)
        
        # channel shuffle
        bs, c, h, w = ref_bn2.shape
        cpg = c // 3
        ref_shuffle = ref_bn2.view(bs, 3, cpg, h, w).transpose(1, 2).contiguous().view(bs, c, h, w)
        
        # conv3: groups=3
        ref_conv3 = F.conv2d(ref_shuffle, p_conv3_weight, bias=None, stride=1, padding=0, groups=3)
        ref_bn3 = F.batch_norm(ref_conv3, b_bn3_running_mean, b_bn3_running_var,
                               p_bn3_weight, p_bn3_bias, training=False, eps=1e-5)
        ref_relu3 = F.relu(ref_bn3)
        
        # shortcut
        ref_sc = F.conv2d(x, p_shortcut_0_weight, bias=None, stride=1, padding=0, groups=1)
        ref_sc_bn = F.batch_norm(ref_sc, b_shortcut_1_running_mean, b_shortcut_1_running_var,
                                 p_shortcut_1_weight, p_shortcut_1_bias, training=False, eps=1e-5)
        
        ref_out = ref_relu3 + ref_sc_bn
        
        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref_out).abs().max().item():.4e}")
        assert torch.allclose(output, ref_out, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref_out).abs().max().item()):.4e}"
        print("Kernel executed successfully.")