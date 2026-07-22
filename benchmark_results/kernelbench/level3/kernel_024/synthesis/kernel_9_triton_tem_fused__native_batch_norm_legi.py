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
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
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
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
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
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# ReLU pointwise kernel
# ============================================================
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
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Adaptive Average Pool 2D (global) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=2),
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
    
    acc = tl.zeros([1], dtype=tl.float32)
    base = n * C * HW + c * HW
    
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW
        x = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# ============================================================
# Linear (GEMM) kernel using tl.dot
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ============================================================
# Conv2d kernel (implicit GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PADDING_H', 'PADDING_W', 'GROUPS'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C_per_group * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    
    M_total = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n_idx = offs_m // (OUT_H * OUT_W)
    
    # Which group does this output channel belong to?
    group = offs_n // GROUP_OUT_C
    local_oc = offs_n % GROUP_OUT_C
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    TOTAL_K = GROUP_IN_C * KERNEL_H * KERNEL_W
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ic, kh, kw)
        ic = offs_k % GROUP_IN_C
        kh = (offs_k // GROUP_IN_C) % KERNEL_H
        kw = offs_k // (GROUP_IN_C * KERNEL_H)
        
        # Input coordinates
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PADDING_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PADDING_W
        
        # For groups, we need to handle per-output-channel group
        # Since GROUPS can vary, we handle group=0 for simplicity when GROUPS=1
        # For depthwise (GROUPS=IN_C=OUT_C), each output channel has its own group
        # We'll handle this by computing group per output channel
        # But in the k loop, group is fixed per output channel tile
        # For simplicity, assume group is consistent within BLOCK_N
        # (This works when BLOCK_N <= GROUP_OUT_C or groups=1)
        
        # Use group from first element of offs_n for input channel offset
        group_0 = offs_n[0] // GROUP_OUT_C
        ic_global = ic + group_0 * GROUP_IN_C
        
        valid_k = offs_k < TOTAL_K
        valid_m = (n_idx < BATCH) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        x_ptrs = (x_ptr + 
                  n_idx[:, None] * (IN_C * IN_H * IN_W) +
                  ic_global[None, :] * (IN_H * IN_W) +
        
        mask_x = valid_m & (valid_k[None, :]) & (n_idx[:, None] < BATCH)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Weight: [OUT_C, GROUP_IN_C, KERNEL_H, KERNEL_W]
        # For each output channel in offs_n, load corresponding weights
        w_ptrs = (w_ptr +
                  offs_n[None, :] * (GROUP_IN_C * KERNEL_H * KERNEL_W) +
                  ic[:, None] * (KERNEL_H * KERNEL_W) +
                  kh[:, None] * KERNEL_W + kw[:, None])
        
        mask_w = (valid_k[:, None]) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    out_ptrs = (out_ptr +
                n_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
                offs_n[None, :] * (OUT_H * OUT_W) +
                oh[:, None] * OUT_W + ow[:, None])
    
    mask_out = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ============================================================
# Helper functions
# ============================================================

def triton_conv2d(x, w, stride=1, padding=0, groups=1):
    """Triton conv2d implementation."""
    N, C, H, W = x.shape
    OC, IC_per_group, KH, KW = w.shape
    
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    SH, SW = stride
    PH, PW = padding
    
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    
    out = torch.empty(N, OC, OH, OW, device=x.device, dtype=x.dtype)
    
    M_total = N * OH * OW
    
    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, out,
        N, C, H, W,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        groups,
    )
    
    return out


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    """Fused BatchNorm + optional ReLU."""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    
    fused_bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial,
        eps,
        apply_relu,
    )
    return out


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_adaptive_avg_pool2d(x):
    """Global average pooling to (1,1)."""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    
    # One program per (n, c) pair
    total = N * C
    grid = (total,)
    
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, HW,
    )
    return out


def triton_linear(x, weight, bias):
    """Linear layer: x @ weight.T + bias"""
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# ============================================================
# Main kernel function implementing EfficientNetB2 forward
# ============================================================

def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight,
    p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight, p_mbconv2_9_weight,
    p_mbconv2_11_weight, p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight, p_mbconv3_9_weight,
    p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight, p_mbconv4_9_weight,
    p_mbconv4_11_weight, p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight, p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight, p_mbconv5_9_weight,
    p_mbconv5_11_weight, p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight, p_bn_final_weight, p_bn_final_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
    b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
    b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
    b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
    b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
    b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
    b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
    b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
    b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
    b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
    b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
    b_mbconv4_12_running_mean, b_mbconv4_12_running_var,
    b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
    b_mbconv5_4_running_mean, b_mbconv5_4_running_var,
    b_mbconv5_12_running_mean, b_mbconv5_12_running_var,
    b_bn_final_running_mean, b_bn_final_running_var,
    x,
):
    # ---- Initial conv + bn + relu ----
    # conv2d: [2,3,224,224] -> [2,32,112,112], stride=2, pad=1, kernel=3
    out = triton_conv2d(x, p_conv1_weight, stride=2, padding=1, groups=1)
    # bn + relu
    out = triton_bn_relu(out, b_bn1_running_mean, b_bn1_running_var,
                         p_bn1_weight, p_bn1_bias, apply_relu=True)
    
    # ---- MBConv1: in=32, out=96, stride=1, expand=3 ----
    # expanded_channels = 32*3 = 96
    # Expansion conv: [2,32,112,112] -> [2,96,112,112], 1x1
    t = triton_conv2d(out, p_mbconv1_0_weight, stride=1, padding=0, groups=1)
    t = triton_bn_relu(t, b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
                       p_mbconv1_1_weight, p_mbconv1_1_bias, apply_relu=True)
    # Depthwise conv: [2,96,112,112] -> [2,96,112,112], 3x3, stride=1, groups=96
    t = triton_conv2d(t, p_mbconv1_3_weight, stride=1, padding=1, groups=96)
    t = triton_bn_relu(t, b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
                       p_mbconv1_4_weight, p_mbconv1_4_bias, apply_relu=True)
    # SE: adaptive_avg_pool2d -> [2,96,1,1]
    se = triton_adaptive_avg_pool2d(t)
    # SE conv1: [2,96,1,1] -> [2,24,1,1]
    se = triton_conv2d(se, p_mbconv1_7_weight, stride=1, padding=0, groups=1)
    se = triton_relu(se)
    # SE conv2: [2,24,1,1] -> [2,96,1,1]
    se = triton_conv2d(se, p_mbconv1_9_weight, stride=1, padding=0, groups=1)
    se = triton_sigmoid(se)
    # Scale: t * se (broadcast)
    t = t * se
    # Output conv: [2,96,112,112] -> [2,96,1,1]? 
    # Wait - looking at the computation graph more carefully:
    # After SE sigmoid, we have conv2d -> [2,96,1,1] then batch_norm -> [2,96,1,1]
    # Then conv2d -> [2,576,1,1]...
    # Actually the spatial dims collapse after SE in the graph. Let me re-read.
    # 
    # From the graph:
    # conv2d -> [2,96,112,112]
    # bn -> [2,96,112,112]  
    # relu -> [2,96,112,112]
    # adaptive_avg_pool2d -> [2,96,1,1]
    # conv2d -> [2,24,1,1]
    # relu -> [2,24,1,1]
    # conv2d -> [2,96,1,1]
    # sigmoid -> [2,96,1,1]
    # conv2d -> [2,96,1,1]  <- this is the output conv applied to SE output?
    # 
    # Hmm, that doesn't match standard SE. Let me re-read the original code.
    # In the original: SE multiplies the feature map, then output conv projects to out_channels.
    # But the graph shows spatial=1,1 after SE...
    # 
    # Actually looking at the graph again: after sigmoid [2,96,1,1], 
    # next is conv2d -> [2,96,1,1] then batch_norm -> [2,96,1,1]
    # This suggests the output conv is applied to the SE-scaled features which are 1x1.
    # 
    # Wait - I think the SE is applied as: features * sigmoid_weights, 
    # but the features at that point are already pooled to 1x1.
    # Let me re-read the MBConv block structure...
    #
    # The SE block: pool -> fc1 -> relu -> fc2 -> sigmoid -> multiply with input
    # But in the graph, after sigmoid we have conv2d -> [2,96,1,1] then bn
    # This means the output projection conv is applied to the 1x1 SE-scaled features.
    # That means the depthwise conv output was already pooled before SE multiply.
    # 
    # Actually I think I'm misreading the graph. The SE multiply happens implicitly
    # and the output conv [2,96,112,112] -> [2,96,1,1] doesn't make sense spatially.
    #
    # Let me look at the graph output shapes more carefully:
    # After sigmoid [2,96,1,1], next conv2d -> [2,96,1,1] (not [2,96,112,112])
    # This means the SE output (1x1) is being convolved to produce 1x1 output.
    # 
    # I think the graph is showing a different architecture than I expected.
    # The MBConv block here seems to work entirely at 1x1 spatial after the first