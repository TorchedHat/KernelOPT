import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: Fused BN + ReLU6 (pointwise) ───────────────────────────────────
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
def bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index: layout is [N, C, H, W] so channel = (offset // (H*W)) % C
    # But we pass C and numel=N*C*H*W, spatial = numel // C
    # channel index = (offset // spatial_per_channel) % C
    # We'll compute channel from offset
    # offset = n * C * HW + c * HW + hw
    # c = (offset // HW) % C  -- but HW = numel / (N*C) ... we need spatial
    # Pass spatial_size separately
    c_idx = (offsets // (numel // C)) % C
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    out = tl.minimum(out, 6.0)
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
    numel, C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // (numel // C)) % C
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offsets, out, mask=mask)


# ─── Kernel 2: 1x1 Conv (GEMM) ────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ─── Kernel 3: Depthwise Conv2d 5x5 stride=2 pad=2 ───────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C': 1}, num_warps=2),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv5x5_s2_p2_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles BLOCK_M output spatial positions for one channel
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    total_spatial = N * OUT_H * OUT_W
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < total_spatial
    
    # Decode spatial position
    n = offs_m // (OUT_H * OUT_W)
    rem = offs_m % (OUT_H * OUT_W)
    oh = rem // OUT_W
    ow = rem % OUT_W
    
    c = pid_c  # single channel per program
    
    # Load weights for this channel (5x5 = 25 values)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for kh in range(5):
        for kw in range(5):
            ih = oh * 2 + kh - 2
            iw = ow * 2 + kw - 2
            valid = mask_m & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            w_idx = c * 25 + kh * 5 + kw
            w_val = tl.load(w_ptr + w_idx, mask=mask_m, other=0.0)  # broadcast
            acc += x_val * w_val
    
    out_idx = n * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W + ow
    tl.store(out_ptr + out_idx, acc, mask=mask_m)


def kernel_function(
):
    N, C_in, H, W = x.shape  # 10, 112, 224, 224
    C_hidden = 672
    C_out = 192
    device = x.device
    eps = 1e-5

    # ── Step 1: 1x1 Conv (expand): [10,112,224,224] -> [10,672,224,224] ──────
    # Reshape x to [N*H*W, C_in] = [501760, 112]
    # Weight: [672, 112] (after squeeze)
    x_2d = x.permute(0, 2, 3, 1).contiguous().reshape(-1, C_in)  # [501760, 112]
    w_expand = p_expand_conv_0_weight.squeeze(-1).squeeze(-1)  # [672, 112]
    
    M1 = x_2d.shape[0]  # 501760
    N1 = C_hidden        # 672
    K1 = C_in            # 112
    
    expand_out = torch.empty((M1, N1), device=device, dtype=torch.float32)
    
    grid1 = (triton.cdiv(M1, 128), triton.cdiv(N1, 64))
    conv1x1_kernel[grid1](
        x_2d, w_expand, expand_out,
        M1, N1, K1,
        x_2d.stride(0), x_2d.stride(1),
        w_expand.stride(0), w_expand.stride(1),
        expand_out.stride(0), expand_out.stride(1),
    )
    # Reshape to [N, H, W, C_hidden] then to [N, C_hidden, H, W]
    expand_out_4d = expand_out.reshape(N, H, W, C_hidden).permute(0, 3, 1, 2).contiguous()
    
    # ── Step 2: BN + ReLU6 ──────────────────────────────────────────────────
    numel1 = expand_out_4d.numel()  # 10*672*224*224
    bn1_out = torch.empty_like(expand_out_4d)
    grid_bn1 = (triton.cdiv(numel1, 1024),)
    bn_relu6_kernel[grid_bn1](
        expand_out_4d, bn1_out,
        b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
        p_expand_conv_1_weight, p_expand_conv_1_bias,
        numel1, C_hidden,
        eps,
    )
    
    # ── Step 3: Depthwise Conv 5x5 stride=2 pad=2 ───────────────────────────
    OUT_H2 = (H + 2*2 - 5) // 2 + 1  # 112
    OUT_W2 = (W + 2*2 - 5) // 2 + 1  # 112
    dw_out = torch.empty((N, C_hidden, OUT_H2, OUT_W2), device=device, dtype=torch.float32)
    w_dw = p_depthwise_conv_0_weight.reshape(C_hidden, 25)  # [672, 25]
    
    total_spatial2 = N * OUT_H2 * OUT_W2
    grid_dw = (triton.cdiv(total_spatial2, 128), C_hidden)
    depthwise_conv5x5_s2_p2_kernel[grid_dw](
        bn1_out, w_dw, dw_out,
        N, C_hidden, H, W, OUT_H2, OUT_W2,
    )
    
    # ── Step 4: BN + ReLU6 ──────────────────────────────────────────────────
    numel2 = dw_out.numel()
    bn2_out = torch.empty_like(dw_out)
    grid_bn2 = (triton.cdiv(numel2, 1024),)
    bn_relu6_kernel[grid_bn2](
        dw_out, bn2_out,
        b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
        p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
        numel2, C_hidden,
        eps,
    )
    
    # ── Step 5: 1x1 Conv (project): [10,672,112,112] -> [10,192,112,112] ────
    x_2d2 = bn2_out.permute(0, 2, 3, 1).contiguous().reshape(-1, C_hidden)  # [125440, 672]
    w_proj = p_project_conv_0_weight.squeeze(-1).squeeze(-1)  # [192, 672]
    
    M3 = x_2d2.shape[0]  # 125440
    N3 = C_out            # 192
    K3 = C_hidden         # 672
    
    proj_out = torch.empty((M3, N3), device=device, dtype=torch.float32)
    grid3 = (triton.cdiv(M3, 128), triton.cdiv(N3, 64))
    conv1x1_kernel[grid3](
        x_2d2, w_proj, proj_out,
        M3, N3, K3,
        x_2d2.stride(0), x_2d2.stride(1),
        w_proj.stride(0), w_proj.stride(1),
        proj_out.stride(0), proj_out.stride(1),
    )
    proj_out_4d = proj_out.reshape(N, OUT_H2, OUT_W2, C_out).permute(0, 3, 1, 2).contiguous()
    
    # ── Step 6: BN (no activation) ──────────────────────────────────────────
    numel3 = proj_out_4d.numel()
    bn3_out = torch.empty_like(proj_out_4d)
    grid_bn3 = (triton.cdiv(numel3, 1024),)
    bn_kernel[grid_bn3](
        proj_out_4d, bn3_out,
        b_project_conv_1_running_mean, b_project_conv_1_running_var,
        p_project_conv_1_weight, p_project_conv_1_bias,
        numel3, C_out,
        eps,
    )
    
    return bn3_out


def get_inputs():
    batch_size = 10
    in_channels = 112
    out_channels = 192
    kernel_size = 5
    stride = 2
    expand_ratio = 6
    hidden_dim = in_channels * expand_ratio  # 672

    p_expand_conv_0_weight = torch.randn(hidden_dim, in_channels, 1, 1)
    p_expand_conv_1_weight = torch.ones(hidden_dim)
    p_expand_conv_1_bias = torch.zeros(hidden_dim)
    p_depthwise_conv_0_weight = torch.randn(hidden_dim, 1, kernel_size, kernel_size)
    p_depthwise_conv_1_weight = torch.ones(hidden_dim)
    p_depthwise_conv_1_bias = torch.zeros(hidden_dim)
    p_project_conv_0_weight = torch.randn(out_channels, hidden_dim, 1, 1)
    p_project_conv_1_weight = torch.ones(out_channels)
    p_project_conv_1_bias = torch.zeros(out_channels)
    b_expand_conv_1_running_mean = torch.zeros(hidden_dim)
    b_expand_conv_1_running_var = torch.ones(hidden_dim)
    b_depthwise_conv_1_running_mean = torch.zeros(hidden_dim)
    b_depthwise_conv_1_running_var = torch.ones(hidden_dim)
    b_project_conv_1_running_mean = torch.zeros(out_channels)
    b_project_conv_1_running_var = torch.ones(out_channels)
    x = torch.rand(batch_size, in_channels, 224, 224)

    return [
        p_expand_conv_0_weight,
        p_expand_conv_1_weight,
        p_expand_conv_1_bias,
        p_depthwise_conv_0_weight,
        p_depthwise_conv_1_weight,
        p_depthwise_conv_1_bias,
        p_project_conv_0_weight,
        p_project_conv_1_weight,
        p_project_conv_1_bias,
        b_expand_conv_1_running_mean,
        b_expand_conv_1_running_var,
        b_depthwise_conv_1_running_mean,
        b_depthwise_conv_1_running_var,
        b_project_conv_1_running_mean,
        b_project_conv_1_running_var,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        import torch.nn as nn

        batch_size = 10
        in_channels = 112
        out_channels = 192
        kernel_size = 5
        stride = 2
        expand_ratio = 6

        (p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
         p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
         p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
         b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
         b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
         b_project_conv_1_running_mean, b_project_conv_1_running_var,
         x) = inputs_cuda

        hidden_dim = in_channels * expand_ratio  # 672

        # Expand conv + BN + ReLU6
        x1 = torch.nn.functional.conv2d(x, p_expand_conv_0_weight, bias=None, stride=1, padding=0)
        x1 = (x1 - b_expand_conv_1_running_mean[None, :, None, None]) / torch.sqrt(b_expand_conv_1_running_var[None, :, None, None] + 1e-5)
        x1 = x1 * p_expand_conv_1_weight[None, :, None, None] + p_expand_conv_1_bias[None, :, None, None]
        x1 = x1.clamp(0, 6)

        # Depthwise conv + BN + ReLU6
        x2 = torch.nn.functional.conv2d(x1, p_depthwise_conv_0_weight, bias=None, stride=2, padding=2, groups=hidden_dim)
        x2 = (x2 - b_depthwise_conv_1_running_mean[None, :, None, None]) / torch.sqrt(b_depthwise_conv_1_running_var[None, :, None, None] + 1e-5)
        x2 = x2 * p_depthwise_conv_1_weight[None, :, None, None] + p_depthwise_conv_1_bias[None, :, None, None]
        x2 = x2.clamp(0, 6)

        # Project conv + BN
        x3 = torch.nn.functional.conv2d(x2, p_project_conv_0_weight, bias=None, stride=1, padding=0)
        x3 = (x3 - b_project_conv_1_running_mean[None, :, None, None]) / torch.sqrt(b_project_conv_1_running_var[None, :, None, None] + 1e-5)
        x3 = x3 * p_project_conv_1_weight[None, :, None, None] + p_project_conv_1_bias[None, :, None, None]

        ref = x3

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")