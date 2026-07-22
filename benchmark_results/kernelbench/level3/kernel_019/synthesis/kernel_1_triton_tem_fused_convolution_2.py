import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    num_channels,
    spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) -> channel = (offset / spatial) % C
    channel = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel, mask=mask, other=0.0)
    w = tl.load(weight_ptr + channel, mask=mask, other=1.0)
    b = tl.load(bias_ptr + channel, mask=mask, other=0.0)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * w + b

    # ReLU
    out = tl.maximum(x_norm, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var,
        weight, bias,
        numel, C, spatial, eps,
    )
    return out


# Fused AvgPool + Flatten + Linear kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def avgpool_linear_kernel(
    in_channels, # C = K
    M,           # batch = N
    K,           # in_features = C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    spatial = pool_h * pool_w
    inv_spatial = 1.0 / (pool_h * pool_w).to(tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load and average pool: x has shape (N, C, H, W)
        # For each (batch_idx, channel_idx), average over H*W
        # x_pooled[n, c] = mean over spatial of x[n, c, :, :]
        # We load x[offs_m, offs_k, :, :] and average
        x_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for s in range(0, spatial, 1):
            # x[n, c, h, w] at linear index: n * C*H*W + c * H*W + s
            x_idx = (offs_m[:, None] * (in_channels * spatial) +
                     offs_k[None, :] * spatial + s)
            x_mask = (offs_m[:, None] < batch) & (offs_k[None, :] < K)
            x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
            x_tile = x_tile + x_val

        x_tile = x_tile * inv_spatial

        # Load weight tile: w[offs_n, offs_k] -> shape (BLOCK_N, BLOCK_K) -> transpose to (BLOCK_K, BLOCK_N)
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b_mask = offs_n < N
    bias = tl.load(b_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store output
    out_idx = offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def kernel_function(
    p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
    p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
    p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
    p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
    p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
    p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
    p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
    p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
    p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
    p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
    p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
    p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
    p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
    p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
    p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
    p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
    p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
    p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
    p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
    p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
    p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
    p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
    p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
    p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
    p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
    p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
    p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
    p_fc_weight, p_fc_bias,
    b_model_0_1_running_mean, b_model_0_1_running_var,
    b_model_1_1_running_mean, b_model_1_1_running_var,
    b_model_1_4_running_mean, b_model_1_4_running_var,
    b_model_2_1_running_mean, b_model_2_1_running_var,
    b_model_2_4_running_mean, b_model_2_4_running_var,
    b_model_3_1_running_mean, b_model_3_1_running_var,
    b_model_3_4_running_mean, b_model_3_4_running_var,
    b_model_4_1_running_mean, b_model_4_1_running_var,
    b_model_4_4_running_mean, b_model_4_4_running_var,
    b_model_5_1_running_mean, b_model_5_1_running_var,
    b_model_5_4_running_mean, b_model_5_4_running_var,
    b_model_6_1_running_mean, b_model_6_1_running_var,
    b_model_6_4_running_mean, b_model_6_4_running_var,
    b_model_7_1_running_mean, b_model_7_1_running_var,
    b_model_7_4_running_mean, b_model_7_4_running_var,
    b_model_8_1_running_mean, b_model_8_1_running_var,
    b_model_8_4_running_mean, b_model_8_4_running_var,
    b_model_9_1_running_mean, b_model_9_1_running_var,
    b_model_9_4_running_mean, b_model_9_4_running_var,
    b_model_10_1_running_mean, b_model_10_1_running_var,
    b_model_10_4_running_mean, b_model_10_4_running_var,
    b_model_11_1_running_mean, b_model_11_1_running_var,
    b_model_11_4_running_mean, b_model_11_4_running_var,
    b_model_12_1_running_mean, b_model_12_1_running_var,
    b_model_12_4_running_mean, b_model_12_4_running_var,
    b_model_13_1_running_mean, b_model_13_1_running_var,
    b_model_13_4_running_mean, b_model_13_4_running_var,
    x,
):
    # Layer 0: conv_bn (3->32, stride=2)
    out = F.conv2d(x, p_model_0_0_weight, None, stride=2, padding=1)
    out = fused_bn_relu(out, p_model_0_1_weight, p_model_0_1_bias,

    # Layer 1: conv_dw (32->32 dw, stride=1) + (32->64 pw)
    out = F.conv2d(out, p_model_1_0_weight, None, stride=1, padding=1, groups=32)
    out = fused_bn_relu(out, p_model_1_1_weight, p_model_1_1_bias,
    out = F.conv2d(out, p_model_1_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_1_4_weight, p_model_1_4_bias,

    # Layer 2: conv_dw (64->64 dw, stride=2) + (64->128 pw)
    out = F.conv2d(out, p_model_2_0_weight, None, stride=2, padding=1, groups=64)
    out = fused_bn_relu(out, p_model_2_1_weight, p_model_2_1_bias,
    out = F.conv2d(out, p_model_2_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_2_4_weight, p_model_2_4_bias,

    # Layer 3: conv_dw (128->128 dw, stride=1) + (128->128 pw)
    out = F.conv2d(out, p_model_3_0_weight, None, stride=1, padding=1, groups=128)
    out = fused_bn_relu(out, p_model_3_1_weight, p_model_3_1_bias,
    out = F.conv2d(out, p_model_3_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_3_4_weight, p_model_3_4_bias,

    # Layer 4: conv_dw (128->128 dw, stride=2) + (128->256 pw)
    out = F.conv2d(out, p_model_4_0_weight, None, stride=2, padding=1, groups=128)
    out = fused_bn_relu(out, p_model_4_1_weight, p_model_4_1_bias,
    out = F.conv2d(out, p_model_4_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_4_4_weight, p_model_4_4_bias,

    # Layer 5: conv_dw (256->256 dw, stride=1) + (256->256 pw)
    out = F.conv2d(out, p_model_5_0_weight, None, stride=1, padding=1, groups=256)
    out = fused_bn_relu(out, p_model_5_1_weight, p_model_5_1_bias,
    out = F.conv2d(out, p_model_5_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_5_4_weight, p_model_5_4_bias,

    # Layer 6: conv_dw (256->256 dw, stride=2) + (256->512 pw)
    out = F.conv2d(out, p_model_6_0_weight, None, stride=2, padding=1, groups=256)
    out = fused_bn_relu(out, p_model_6_1_weight, p_model_6_1_bias,
    out = F.conv2d(out, p_model_6_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_6_4_weight, p_model_6_4_bias,

    # Layer 7: conv_dw (512->512 dw, stride=1) + (512->512 pw)
    out = F.conv2d(out, p_model_7_0_weight, None, stride=1, padding=1, groups=512)
    out = fused_bn_relu(out, p_model_7_1_weight, p_model_7_1_bias,
    out = F.conv2d(out, p_model_7_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_7_4_weight, p_model_7_4_bias,

    # Layer 8: conv_dw (512->512 dw, stride=1) + (512->512 pw)
    out = F.conv2d(out, p_model_8_0_weight, None, stride=1, padding=1, groups=512)
    out = fused_bn_relu(out, p_model_8_1_weight, p_model_8_1_bias,
    out = F.conv2d(out, p_model_8_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_8_4_weight, p_model_8_4_bias,

    # Layer 9: conv_dw (512->512 dw, stride=1) + (512->512 pw)
    out = F.conv2d(out, p_model_9_0_weight, None, stride=1, padding=1, groups=512)
    out = fused_bn_relu(out, p_model_9_1_weight, p_model_9_1_bias,
    out = F.conv2d(out, p_model_9_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_9_4_weight, p_model_9_4_bias,

    # Layer 10: conv_dw (512->512 dw, stride=1) + (512->512 pw)
    out = F.conv2d(out, p_model_10_0_weight, None, stride=1, padding=1, groups=512)
    out = fused_bn_relu(out, p_model_10_1_weight, p_model_10_1_bias,
    out = F.conv2d(out, p_model_10_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_10_4_weight, p_model_10_4_bias,

    # Layer 11: conv_dw (512->512 dw, stride=1) + (512->512 pw)
    out = F.conv2d(out, p_model_11_0_weight, None, stride=1, padding=1, groups=512)
    out = fused_bn_relu(out, p_model_11_1_weight, p_model_11_1_bias,
    out = F.conv2d(out, p_model_11_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_11_4_weight, p_model_11_4_bias,

    # Layer 12: conv_dw (512->512 dw, stride=2) + (512->1024 pw)
    out = F.conv2d(out, p_model_12_0_weight, None, stride=2, padding=1, groups=512)
    out = fused_bn_relu(out, p_model_12_1_weight, p_model_12_1_bias,
    out = F.conv2d(out, p_model_12_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_12_4_weight, p_model_12_4_bias,

    # Layer 13: conv_dw (1024->1024 dw, stride=1) + (1024->1024 pw)
    out = F.conv2d(out, p_model_13_0_weight, None, stride=1, padding=1, groups=1024)
    out = fused_bn_relu(out, p_model_13_1_weight, p_model_13_1_bias,
    out = F.conv2d(out, p_model_13_3_weight, None, stride=1, padding=0)
    out = fused_bn_relu(out, p_model_13_4_weight, p_model_13_4_bias,

    # AvgPool + flatten + linear using Triton
    N_batch, C, H, W = out.shape
    out_fc = torch.empty((N_batch, 1000), device=out.device, dtype=out.dtype)
    M_val = N_batch
    N_val = 1000
    K_val = C

    grid = lambda meta: (
        triton.cdiv(M_val, meta['BLOCK_M']),
        triton.cdiv(N_val, meta['BLOCK_N']),
    )
    avgpool_linear_kernel[grid](
        out, p_fc_weight, p_fc_bias, out_fc,
        N_batch, C, H, W,
        M_val, N_val, K_val,
    )
    return out_fc


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'

    def make_tensor(*shape):
        return torch.randn(*shape, device=device)

    def make_pos(*shape):
        return torch.abs(torch.randn(*shape, device=device)) + 0.1

    inputs = [
        make_tensor(32, 3, 3, 3),   # p_model_0_0_weight
        make_tensor(32),             # p_model_0_1_weight
        make_tensor(32),             # p_model_0_1_bias
        make_tensor(32, 1, 3, 3),   # p_model_1_0_weight
        make_tensor(32),             # p_model_1_1_weight
        make_tensor(32),             # p_model_1_1_bias
        make_tensor(64, 32, 1, 1),  # p_model_1_3_weight
        make_tensor(64),             # p_model_1_4_weight
        make_tensor(64),             # p_model_1_4_bias
        make_tensor(64, 1, 3, 3),   # p_model_2_0_weight
        make_tensor(64),             # p_model_2_1_weight
        make_tensor(64),             # p_model_2_1_bias
        make_tensor(128, 64, 1, 1), # p_model_2_3_weight
        make_tensor(128),            # p_model_2_4_weight
        make_tensor(128),            # p_model_2_4_bias
        make_tensor(128, 1, 3, 3),  # p_model_3_0_weight
        make_tensor(128),            # p_model_3_1_weight
        make_tensor(128),            # p_model_3_1_bias
        make_tensor(128, 128, 1, 1),# p_model_3_3_weight
        make_tensor(128),            # p_model_3_4_weight
        make_tensor(128),            # p_model_3_4_bias
        make_tensor(128, 1, 3, 3),  # p_model_4_0_weight
        make_tensor(128),            # p_model_4_1_weight
        make_tensor(128),            # p_model_4_1_bias
        make_tensor(256, 128, 1, 1),# p_model_4_3_weight
        make_tensor(256),            # p_model_4_4_weight
        make_tensor(256),            # p_model_4_4_bias
        make_tensor(256, 1, 3, 3),  # p_model_5_0_weight
        make_tensor(256),            # p_model_5_1_weight
        make_tensor(256),            # p_model_5_1_bias
        make_tensor(256, 256, 1, 1),# p_model_5_3_weight
        make_tensor(256),            # p_model_5_4_weight
        make_tensor(256),            # p_model_5_4_bias
        make_tensor(256, 1, 3, 3),  # p_model_6_0_weight
        make_tensor(256),            # p_model_6_1_weight
        make_tensor(256),            # p_model_6_1_bias
        make_tensor(512, 256, 1, 1),# p_model_6_3_weight
        make_tensor(512),            # p_model_6_4_weight
        make_tensor(512),            # p_model_6_4_bias
        make_tensor(512, 1, 3, 3),  # p_model_7_0_weight
        make_tensor(512),            # p_model_7_1_weight
        make_tensor(512),            # p_model_7_1_bias
        make_tensor(512, 512, 1, 1),# p_model_7_3_weight
        make_tensor(512),            # p_model_7_4_weight
        make_tensor(512),            # p_model_7_4_bias
        make_tensor(512, 1, 3, 3),  # p_model_8_0_weight
        make_tensor(512),            # p_model_8_1_weight
        make_tensor(512),            # p_model_8_1_bias
        make_tensor(512, 512, 1, 1),# p_model_8_3_weight
        make_tensor(512),            # p_model_8_4_weight
        make_tensor(512),            # p_model_8_4_bias
        make_tensor(512, 1, 3, 3),  # p_model_9_0_weight
        make_tensor(512),            # p_model_9_1_weight
        make_tensor(512),            # p_model_9_1_bias
        make_tensor(512, 512, 1, 1),# p_model_9_3_weight
        make_tensor(512),            # p_model_9_4_weight
        make_tensor(512),            # p_model_9_4_bias
        make_tensor(512,