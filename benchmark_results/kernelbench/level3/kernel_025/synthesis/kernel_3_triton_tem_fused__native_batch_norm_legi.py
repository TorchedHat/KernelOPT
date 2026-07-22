import torch
import triton
import triton.language as tl
import math

# ============================================================
# Kernel 1: Fused BN + ReLU (pointwise epilogue)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index: layout is (N, C, H, W) contiguous => channel = (offset // (H*W)) % C
    # But we need to know HW. We pass C and compute c = offset % C for NHWC or
    # For NCHW: c = (offset // HW) % C. We'll pass HW as a param.
    # Actually let's just pass C and HW separately.
    # We'll restructure: pass spatial = H*W
    c_idx = (offsets // 1) % C  # placeholder, will fix below
    # This kernel is called with proper c_idx computation
    # Let's use a different approach: pass stride_c
    tl.store(out_ptr + offsets, x, mask=mask)


# Better approach: separate kernel with explicit channel stride
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_nchw_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, HW, C,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    c_idx = (flat_idx // HW) % C
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if apply_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Kernel 2: Channel Shuffle
# Input: (N, C, H, W) contiguous, groups=G
# Output: (N, C, H, W) after shuffle
# Shuffle: view(N, G, C//G, H, W) -> transpose(1,2) -> view(N, C, H, W)
# So output[n, c, h, w] = input[n, (c % G) * (C//G) + c // G, h, w]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    numel, C, HW, G,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Channel shuffle: groups=G, C channels
    out[n, c, h, w] = in[n, (c % G) * (C//G) + c // G, h, w]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode (n, c, hw) from flat index
    hw_idx = offsets % HW
    c_idx = (offsets // HW) % C
    n_idx = offsets // (HW * C)

    # Compute source channel
    CPG = C // G  # channels per group
    src_c = (c_idx % G) * CPG + (c_idx // G)

    src_offset = n_idx * C * HW + src_c * HW + hw_idx
    src_mask = mask & (src_offset < numel)

    x = tl.load(x_ptr + src_offset, mask=src_mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


# ============================================================
# Kernel 3: Fused BN + Add (for final add with shortcut)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_add_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    residual_ptr,
    out_ptr,
    numel, HW, C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    BN(x) + residual, no relu
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res

    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Group Convolution 1x1 using tl.dot
# Input: (N, C_in, H, W), Weight: (C_out, C_in//G, 1, 1)
# groups=G
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def group_conv1x1_kernel(
    X_ptr, W_ptr, Out_ptr,
    # dimensions
    N_batch, C_in, C_out, HW,
    # strides for X: (N, C, H*W) -> X[n, c, hw] = X_ptr[n*C_in*HW + c*HW + hw]
    # strides for W: (C_out, C_in//G, 1, 1) -> W[co, ci, 0, 0] = W_ptr[co*(C_in//G) + ci]
    M, N, K,  # M=N_batch*HW, N=C_out//G, K=C_in//G
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
      Out[n, g*N + on, hw] = sum_k X[n, g*K + k, hw] * W[g*N + on, k]
    
    We launch with grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), G)
    where M = N_batch * HW, N = C_out // G, K = C_in // G
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m -> (n_batch, hw)
    hw_idx = offs_m % HW
    n_idx = offs_m // HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X tile: X[n_idx, pid_g*K + offs_k, hw_idx]
        # X layout: (N_batch, C_in, HW) -> flat index = n_idx*C_in*HW + (pid_g*K+offs_k)*HW + hw_idx
        x_c = pid_g * K + offs_k  # [BLOCK_K]
        x_flat = n_idx[:, None] * C_in * HW + x_c[None, :] * HW + hw_idx[:, None]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(X_ptr + x_flat, mask=x_mask, other=0.0)

        # Load W tile: W[pid_g*N + offs_n, offs_k]
        # W layout: (C_out, K) -> flat = (pid_g*N + offs_n)*K + offs_k
        w_row = pid_g * N + offs_n  # [BLOCK_N]
        w_flat = w_row[None, :] * K + offs_k[:, None]  # [BLOCK_K, BLOCK_N]
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(W_ptr + w_flat, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output: Out[n_idx, pid_g*N + offs_n, hw_idx]
    # Out layout: (N_batch, C_out, HW) -> flat = n_idx*C_out*HW + (pid_g*N+offs_n)*HW + hw_idx
    out_c = pid_g * N + offs_n  # [BLOCK_N]
    out_flat = n_idx[:, None] * C_out * HW + out_c[None, :] * HW + hw_idx[:, None]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + out_flat, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Depthwise Conv 3x3 using tl.dot
# Input: (N, C, H, W), Weight: (C, 1, 3, 3)
# groups = C (depthwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['M_total', 'C'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    X_ptr, W_ptr, Out_ptr,
    N_batch, C, H, W_dim,
    BLOCK_M: tl.constexpr,
):
    """
    Depthwise conv 3x3, padding=1, stride=1
    Out[n, c, h, w] = sum_{kh,kw} X[n, c, h+kh-1, w+kw-1] * W[c, 0, kh, kw]
    
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m -> (n, h, w)
    w_idx = offs_m % W_dim
    h_idx = (offs_m // W_dim) % H
    n_idx = offs_m // (H * W_dim)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over 3x3 kernel
    for kh in range(3):
        for kw in range(3):
            ih = h_idx + kh - 1  # padding=1
            iw = w_idx + kw - 1

            valid_h = (ih >= 0) & (ih < H)
            valid_w = (iw >= 0) & (iw < W_dim)
            valid_m = offs_m < M_total
            valid_c = offs_c < C

            # X[n, c, ih, iw] -> flat = n*C*H*W + c*H*W + ih*W + iw
            x_flat = n_idx[:, None] * C * H * W_dim + offs_c[None, :] * H * W_dim + ih[:, None] * W_dim + iw[:, None]
            x_mask = valid_m[:, None] & valid_c[None, :] & valid_h[:, None] & valid_w[:, None]
            x_val = tl.load(X_ptr + x_flat, mask=x_mask, other=0.0)

            # W[c, 0, kh, kw] -> flat = c*9 + kh*3 + kw
            w_flat = offs_c * 9 + kh * 3 + kw
            w_mask = valid_c
            w_val = tl.load(W_ptr + w_flat, mask=w_mask, other=0.0)

            # Element-wise multiply and accumulate
            acc = acc + x_val * w_val[None, :]

    # Store: Out[n, c, h, w] -> flat = n*C*H*W + c*H*W + h*W + w
    out_flat = n_idx[:, None] * C * H * W_dim + offs_c[None, :] * H * W_dim + h_idx[:, None] * W_dim + w_idx[:, None]
    out_mask = (offs_m[:, None] < M_total) & (offs_c[None, :] < C)
    tl.store(Out_ptr + out_flat, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Main kernel function
# ============================================================
def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
    p_conv3_weight, p_bn3_weight, p_bn3_bias,
    p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    b_bn3_running_mean, b_bn3_running_var,
    b_shortcut_1_running_mean, b_shortcut_1_running_var,
    x,
):
    N, C_in, H, W = x.shape  # (10, 240, 224, 224)
    groups = 3
    mid_channels = 120  # C_out of conv1
    out_channels = 480  # C_out of conv3

    HW = H * W
    M = N * HW
    eps = 1e-5

    device = x.device
    dtype = x.dtype

    # ---- Step 1: Conv1 (group conv 1x1, groups=3) ----
    # Input: (10, 240, 224, 224), Weight: (120, 80, 1, 1)
    # C_in=240, C_out=120, G=3, K=C_in//G=80, N_out=C_out//G=40
    C1_in = C_in  # 240
    C1_out = mid_channels  # 120
    G1 = groups  # 3
    K1 = C1_in // G1  # 80
    N1 = C1_out // G1  # 40

    conv1_out = torch.empty((N, C1_out, H, W), device=device, dtype=dtype)
    # Reshape weight: (120, 80, 1, 1) -> (120, 80)
    w1 = p_conv1_weight.reshape(C1_out, K1)

    grid1 = (
        triton.cdiv(M, 128),
        triton.cdiv(N1, 64),
        G1,
    )
    group_conv1x1_kernel[grid1](
        x, w1, conv1_out,
        N, C1_in, C1_out, HW,
        G1,
        M, N1, K1,
    )

    # ---- Step 2: BN1 + ReLU ----
    bn1_out = torch.empty_like(conv1_out)
    numel1 = conv1_out.numel()
    grid_bn1 = (triton.cdiv(numel1, 1024),)
    fused_bn_relu_nchw_kernel[grid_bn1](
        conv1_out, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias,
        bn1_out,
        numel1, HW, C1_out,
        eps, True,
    )

    # ---- Step 3: Conv2 (depthwise 3x3, groups=120) ----
    # Input: (10, 120, 224, 224), Weight: (120, 1, 3, 3)
    conv2_out = torch.empty((N, mid_channels, H, W), device=device, dtype=dtype)
    M2 = N * HW
    grid2 = (
        triton.cdiv(M2, 128),
        triton.cdiv(mid_channels, 32),
    )
    depthwise_conv3x3_kernel[grid2](
        bn1_out, p_conv2_weight, conv2_out,
        N, mid_channels, H, W,
        M2,
    )

    # ---- Step 4: BN2 (no relu) ----
    bn2_out = torch.empty_like(conv2_out)
    numel2 = conv2_out.numel()
    grid_bn2 = (triton.cdiv(numel2, 1024),)
    fused_bn_relu_nchw_kernel[grid_bn2](
        conv2_out, b_bn2_running_mean, b_bn2_running_var, p_bn2_weight, p_bn2_bias,
        bn2_out,
        numel2, HW, mid_channels,
        eps, False,
    )

    # ---- Step 5: Channel Shuffle (groups=3) ----
    shuffle_out = torch.empty_like(bn2_out)
    numel_s = bn2_out.numel()
    grid_s = (triton.cdiv(numel_s, 1024),)
    channel_shuffle_kernel[grid_s](
        bn2_out, shuffle_out,
        numel_s, mid_channels, HW, groups,
    )

    # ---- Step 6: Conv3 (group conv 1x1, groups=3) ----
    # Input: (10, 120, 224, 224), Weight: (480, 40, 1, 1)
    C3_in = mid_channels  # 120
    C3_out = out_channels  # 480
    G3 = groups  # 3
    K3 = C3_in // G3  # 40
    N3 = C3_out // G3  # 160

    conv3_out = torch.empty((N, C3_out, H, W), device=device, dtype=dtype)
    w3 = p_conv3_weight.reshape(C3_out, K3)

    grid3 = (
        triton.cdiv(M, 128),
        triton.cdiv(N3, 64),
        G3,
    )
    group_conv1x1_kernel[grid3](
        shuffle_out, w3, conv3_out,
        N, C3_in, C3_out, HW,
        G3,
        M, N3, K3,
    )

    # ---- Step 7: BN3 + ReLU ----
    bn3_out = torch.empty_like(conv3_out)
    numel3 = conv3_out.numel()
    grid_bn3 = (triton.cdiv(numel3, 1024),)
    fused_bn_relu_nchw_kernel[grid_bn3](
        conv3_out, b_bn3_running_mean, b_bn3_running_var, p_bn3_weight, p_bn3_bias,
        bn3_out,
        numel3, HW, C3_out,
        eps, True,
    )

    # ---- Step 8: Shortcut Conv (1x1, no groups) ----
    # Input: (10, 240, 224, 224), Weight: (480, 240, 1, 1)
    C_sc_in = C_in  # 240
    C_sc_out = out_channels  # 480
    G_sc = 1
    K_sc = C_sc_in  # 240
    N_sc = C_sc_out  # 480

    shortcut_conv_out = torch.empty((N, C_sc_out, H, W), device=device, dtype=dtype)
    w_sc = p_shortcut_0_weight.reshape(C_sc_out, K_sc)

    # For G=1, use group_conv1x1_kernel with G=1
    grid_sc = (
        triton.cdiv(M, 128),
        triton.cdiv(N_sc, 64),
        1,
    )
    group_conv1x1_kernel[grid_sc](
        x, w_sc, shortcut_conv_out,
        N, C_sc_in, C_sc_out, HW,
        1,
        M, N_sc, K_sc,
    )

    # ---- Step 9: BN shortcut + Add ----
    out = torch.empty_like(bn3_out)
    numel_sc = shortcut_conv_out.numel()
    grid_sc_bn = (triton.cdiv(numel_sc, 1024),)
    fused_bn_add_kernel[grid_sc_bn](
        shortcut_conv_out,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
        p_shortcut_1_weight, p_shortcut_1_bias,
        bn3_out,
        out,
        numel_sc, HW, C_sc_out,
        eps,
    )

    return out


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32

    p_conv1_weight = torch.randn(120, 80, 1, 1, device=device, dtype=dtype)
    p_bn1_weight = torch.ones(120, device=device, dtype=dtype)
    p_bn1_bias = torch.zeros(120, device=device, dtype=dtype)
    p_conv2_weight = torch.randn(120, 1, 3, 3, device=device, dtype=dtype)
    p_bn2_weight = torch.ones(120, device=device, dtype=dtype)
    p_bn2_bias = torch.zeros(120, device=device, dtype=dtype)
    p_conv3_weight = torch.randn(480, 40, 1, 1, device=device, dtype=dtype)
    p_bn3_weight = torch.ones(480, device=device, dtype=dtype)
    p_bn3_bias = torch.zeros(480, device=device, dtype=dtype)
    p_shortcut_0_weight = torch.randn(480, 240, 1, 1, device=device, dtype=dtype)
    p_shortcut_1_weight = torch.ones(480, device=device, dtype=dtype)
    p_shortcut_1_bias = torch.zeros(480, device=device, dtype=dtype)
    b_bn1_running_mean = torch.zeros(120, device=device, dtype=dtype)
    b_bn1_running_var = torch.ones(120, device=device, dtype=dtype)
    b_bn2_running_mean = torch.zeros(120, device=device, dtype=dtype)
    b_bn2_running_var = torch.ones(120, device=device, dtype=dtype)
    b_bn3_running_mean = torch.zeros(480, device=device, dtype=dtype)
    b_bn3_running_var = torch.ones(480, device=device, dtype=dtype)
    b_shortcut_1_running_mean = torch.zeros(480, device=device, dtype=dtype)
    b_shortcut_1_running_var = torch.ones(480, device=device, dtype=dtype)
    x = torch.randn(10, 240, 224, 224, device=device, dtype=dtype)

    return [
        p_conv1_weight, p_bn1_weight, p_bn1_bias,
        p_conv2_weight, p_bn2_weight, p_bn2_bias,
        p_conv3_weight, p_bn3_weight, p_bn3_bias,
        p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
        b_bn1_running_mean, b_bn1_running_var,
        b_bn2_running_mean, b_bn2_running_var,
        b_bn3_running_mean, b_bn3_running_var,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
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

        (p_conv1_weight, p_bn1_weight, p_bn1_bias,
         p_conv2_weight, p_bn2_weight, p_bn2_bias,