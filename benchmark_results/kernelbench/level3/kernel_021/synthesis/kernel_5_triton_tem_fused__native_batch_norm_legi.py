import torch
import triton
import triton.language as tl
import math

# ============================================================
# Kernel 1: 1x1 Conv + BN + ReLU6 (expand: 112->672, spatial 224x224)
# M = 10*224*224 = 501760, N = 672, K = 112
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bn_relu6_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    eps,
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
        # X: [M, K] with strides (stride_xm, stride_xk)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W: [N, K] -> we want [K, N] for matmul
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # BN + ReLU6 epilogue
    m_mask = offs_m < M
    n_mask = offs_n < N

    mean = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=n_mask, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    acc = (acc - mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]
    # ReLU6
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


# ============================================================
# Kernel 2: Depthwise Conv 5x5 stride=2 pad=2 + BN + ReLU6
# Input: [10, 672, 224, 224] -> Output: [10, 672, 112, 112]
# groups=672, so each channel is independent
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 32}, num_stages=2, num_warps=8),
    ],
    key=['C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def depthwise_conv_bn_relu6_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    eps,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # Each program handles BLOCK_C channels and BLOCK_HW output spatial positions
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = offs_c < C
    hw_mask = offs_hw < (OUT_H * OUT_W)

    out_h = offs_hw // OUT_W
    out_w = offs_hw % OUT_W

    # Accumulate over kernel
    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h[None, :] * stride_h + kh - pad_h
            iw = out_w[None, :] * stride_w + kw - pad_w

            valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & hw_mask[None, :] & c_mask[:, None]

            x_idx = pid_n * C * IN_H * IN_W + offs_c[:, None] * IN_H * IN_W + ih * IN_W + iw
            x = tl.load(X_ptr + x_idx, mask=valid, other=0.0)

            # Weight: [C, 1, KH, KW] -> index [c, 0, kh, kw]
            w_idx = offs_c * KH * KW + kh * KW + kw
            w = tl.load(W_ptr + w_idx, mask=c_mask, other=0.0)

            acc += x * w[:, None]

    # BN + ReLU6
    mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0)
    var = tl.load(var_ptr + offs_c, mask=c_mask, other=0.0)
    gamma = tl.load(gamma_ptr + offs_c, mask=c_mask, other=0.0)
    beta = tl.load(beta_ptr + offs_c, mask=c_mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    acc = (acc - mean[:, None]) * inv_std[:, None] * gamma[:, None] + beta[:, None]
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    y_idx = pid_n * C * OUT_H * OUT_W + offs_c[:, None] * OUT_H * OUT_W + out_h[None, :] * OUT_W + out_w[None, :]
    y_mask = c_mask[:, None] & hw_mask[None, :]
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=y_mask)


# ============================================================
# Kernel 3: 1x1 Conv + BN (project: 672->192, spatial 112x112)
# M = 10*112*112 = 125440, N = 192, K = 672
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 192, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bn_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    eps,
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
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # BN epilogue (no activation)
    m_mask = offs_m < M
    n_mask = offs_n < N

    mean = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=n_mask, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    acc = (acc - mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


def kernel_function(
):
    device = x.device
    dtype = x.dtype
    eps = 1e-5

    batch = x.shape[0]
    in_c = x.shape[1]
    in_h = x.shape[2]
    in_w = x.shape[3]

    # ---- Stage 1: Expand 1x1 Conv + BN + ReLU6 ----
    # Input x: [10, 112, 224, 224] -> reshape to [10*224*224, 112]
    # Weight: [672, 112, 1, 1] -> [672, 112]
    expand_C_out = 672
    M1 = batch * in_h * in_w  # 501760
    N1 = expand_C_out          # 672
    K1 = in_c                  # 112

    # Reshape x to [M1, K1]: need [batch*H*W, C] layout
    # x is [N, C, H, W] -> permute to [N, H, W, C] -> reshape [M1, K1]
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [10, 224, 224, 112]
    x_flat = x_nhwc.reshape(M1, K1)  # [501760, 112]

    # Weight: [672, 112] (after squeezing spatial dims)
    w_expand = p_expand_conv_0_weight.reshape(N1, K1)  # [672, 112]

    out1 = torch.empty((M1, N1), device=device, dtype=dtype)

    grid1 = (triton.cdiv(M1, 128), triton.cdiv(N1, 64))
    conv1x1_bn_relu6_kernel[grid1](
        x_flat, w_expand,
        b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
        p_expand_conv_1_weight, p_expand_conv_1_bias,
        out1,
        M1, N1, K1,
        eps,
    )
    # out1: [501760, 672] -> reshape to [10, 224, 224, 672] -> [10, 672, 224, 224]
    out1_nchw = out1.reshape(batch, in_h, in_w, expand_C_out).permute(0, 3, 1, 2).contiguous()

    # ---- Stage 2: Depthwise Conv 5x5 stride=2 pad=2 + BN + ReLU6 ----
    C2 = expand_C_out  # 672
    IN_H2 = in_h       # 224
    IN_W2 = in_w       # 224
    OUT_H2 = (IN_H2 + 2*2 - 5) // 2 + 1  # 112
    OUT_W2 = (IN_W2 + 2*2 - 5) // 2 + 1  # 112
    KH2, KW2 = 5, 5

    out2 = torch.empty((batch, C2, OUT_H2, OUT_W2), device=device, dtype=dtype)

    # Weight: [672, 1, 5, 5] -> [672, 25]
    w_dw = p_depthwise_conv_0_weight.reshape(C2, KH2 * KW2)

    BLOCK_C2 = 32
    BLOCK_HW2 = 64
    grid2 = (
        triton.cdiv(C2, BLOCK_C2),
        triton.cdiv(OUT_H2 * OUT_W2, BLOCK_HW2),
        batch,
    )
    depthwise_conv_bn_relu6_kernel[grid2](
        out1_nchw, p_depthwise_conv_0_weight,
        b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
        p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
        out2,
        batch, C2, IN_H2, IN_W2,
        OUT_H2, OUT_W2,
        KH2, KW2,
        eps,
    )

    # ---- Stage 3: Project 1x1 Conv + BN ----
    proj_C_out = 192
    M3 = batch * OUT_H2 * OUT_W2  # 125440
    N3 = proj_C_out                # 192
    K3 = C2                        # 672

    # out2: [10, 672, 112, 112] -> permute to [10, 112, 112, 672] -> [M3, K3]
    out2_nhwc = out2.permute(0, 2, 3, 1).contiguous()
    out2_flat = out2_nhwc.reshape(M3, K3)

    # Weight: [192, 672] 
    w_proj = p_project_conv_0_weight.reshape(N3, K3)

    out3 = torch.empty((M3, N3), device=device, dtype=dtype)

    grid3 = (triton.cdiv(M3, 128), triton.cdiv(N3, 64))
    conv1x1_bn_kernel[grid3](
        out2_flat, w_proj,
        b_project_conv_1_running_mean, b_project_conv_1_running_var,
        p_project_conv_1_weight, p_project_conv_1_bias,
        out3,
        M3, N3, K3,
        K3, 1,
        K3, 1,
        eps,
    )

    # out3: [125440, 192] -> [10, 112, 112, 192] -> [10, 192, 112, 112]
    out3_nchw = out3.reshape(batch, OUT_H2, OUT_W2, proj_C_out).permute(0, 3, 1, 2).contiguous()

    return out3_nchw


def get_inputs():
    batch_size = 10
    in_channels = 112
    out_channels = 192
    kernel_size = 5
    stride = 2
    expand_ratio = 6
    hidden_dim = in_channels * expand_ratio  # 672

    return [
        torch.randn(hidden_dim, in_channels, 1, 1),   # p_expand_conv_0_weight
        torch.randn(hidden_dim),                        # p_expand_conv_1_weight (gamma)
        torch.randn(hidden_dim),                        # p_expand_conv_1_bias (beta)
        torch.randn(hidden_dim, 1, kernel_size, kernel_size),  # p_depthwise_conv_0_weight
        torch.randn(hidden_dim),                        # p_depthwise_conv_1_weight
        torch.randn(hidden_dim),                        # p_depthwise_conv_1_bias
        torch.randn(out_channels, hidden_dim, 1, 1),   # p_project_conv_0_weight
        torch.randn(out_channels),                      # p_project_conv_1_weight
        torch.randn(out_channels),                      # p_project_conv_1_bias
        torch.zeros(hidden_dim),                        # b_expand_conv_1_running_mean
        torch.ones(hidden_dim),                         # b_expand_conv_1_running_var
        torch.zeros(hidden_dim),                        # b_depthwise_conv_1_running_mean
        torch.ones(hidden_dim),                         # b_depthwise_conv_1_running_var
        torch.zeros(out_channels),                      # b_project_conv_1_running_mean
        torch.ones(out_channels),                       # b_project_conv_1_running_var
        torch.randn(batch_size, in_channels, 224, 224), # x
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    # Move to CUDA
    inputs = [t.cuda() for t in inputs]

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        (p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
         p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
         p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
         b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
         b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
         b_project_conv_1_running_mean, b_project_conv_1_running_var,
         x) = inputs

        # Reference computation
        import torch.nn.functional as F

        # Expand conv + BN + ReLU6
        ref1 = F.conv2d(x, p_expand_conv_0_weight, bias=None, stride=1, padding=0)
        ref1 = F.batch_norm(ref1, b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
                            p_expand_conv_1_weight, p_expand_conv_1_bias, training=False, eps=1e-5)
        ref1 = torch.clamp(ref1, 0.0, 6.0)

        # Depthwise conv + BN + ReLU6
        ref2 = F.conv2d(ref1, p_depthwise_conv_0_weight, bias=None, stride=2, padding=2,
                        groups=672)
        ref2 = F.batch_norm(ref2, b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
                            p_depthwise_conv_1_weight, p_depthwise_conv_1_bias, training=False, eps=1e-5)
        ref2 = torch.clamp(ref2, 0.0, 6.0)

        # Project conv + BN
        ref3 = F.conv2d(ref2, p_project_conv_0_weight, bias=None, stride=1, padding=0)
        ref3 = F.batch_norm(ref3, b_project_conv_1_running_mean, b_project_conv_1_running_var,
                            p_project_conv_1_weight, p_project_conv_1_bias, training=False, eps=1e-5)

        assert torch.allclose(output, ref3, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref3).abs().max().item()):.4e}"
        print("Kernel executed successfully.")