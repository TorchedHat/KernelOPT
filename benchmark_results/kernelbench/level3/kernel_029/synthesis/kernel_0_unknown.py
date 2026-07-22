import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton Kernels
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, N,
    x_stride_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * x_stride_row
    out_row_ptr = out_ptr + row * x_stride_row

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    count = 0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_row_ptr + cols, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_gelu_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, N,
    x_stride_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused LayerNorm + GELU"""
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * x_stride_row
    out_row_ptr = out_ptr + row * x_stride_row

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize, apply weight/bias, then GELU
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = (x - mean) * rstd * w + b
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + tl.extra.libdevice.tanh(0.7978845608028654 * (normed + 0.044715 * normed * normed * normed)))
        out = normed * cdf
        tl.store(out_row_ptr + cols, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cdf = 0.5 * (1.0 + tl.extra.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    """Apply layer norm using Triton kernel"""
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.reshape(M, N)
    out = torch.empty_like(x_2d)
    
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, out, weight, bias,
        M, N,
        x_2d.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


def triton_gelu(x):
    """Apply GELU using Triton kernel"""
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# ============================================================
# Model Components (using Triton for norm/activation)
# ============================================================

def patch_embed_forward(x, proj_weight, proj_bias, norm_weight, norm_bias):
    """Patch embedding: conv2d + flatten + transpose + layer_norm"""
    # Conv2d: (10, 3, 224, 224) -> (10, 96, 56, 56)
    x = torch.nn.functional.conv2d(x, proj_weight, proj_bias, stride=4, padding=0)
    # flatten: (10, 96, 56, 56) -> (10, 96, 3136)
    x = x.flatten(2)
    # transpose: (10, 96, 3136) -> (10, 3136, 96)
    x = x.transpose(1, 2)
    # layer_norm
    x = triton_layer_norm(x, norm_weight, norm_bias)
    return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def swin_mlp_block_forward(x, input_resolution, num_heads, window_size, shift_size,
                            norm1_weight, norm1_bias,
                            spatial_mlp_weight, spatial_mlp_bias,
                            norm2_weight, norm2_bias,
                            mlp_fc1_weight, mlp_fc1_bias,
    """SwinMLPBlock forward pass"""
    H, W = input_resolution
    B, L, C = x.shape
    assert L == H * W

    shortcut = x
    # norm1
    x = triton_layer_norm(x, norm1_weight, norm1_bias)
    x = x.view(B, H, W, C)

    # pad if needed
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    if pad_r > 0 or pad_b > 0:
        x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    _, Hp, Wp, _ = x.shape

    # cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = window_partition(shifted_x, window_size)  # (num_windows*B, ws, ws, C)
    x_windows = x_windows.view(-1, window_size * window_size, C)  # (num_windows*B, ws*ws, C)

    # spatial MLP via conv1d
    # x_windows: (num_windows*B, ws*ws, C)
    # view to (num_windows*B, ws*ws, num_heads, C//num_heads)
    head_dim = C // num_heads
    x_windows = x_windows.view(-1, window_size * window_size, num_heads, head_dim)
    # transpose: (num_windows*B, num_heads, ws*ws, head_dim)
    x_windows = x_windows.transpose(1, 2)
    # reshape: (num_windows*B, num_heads*ws*ws, head_dim)
    num_windows_B = x_windows.shape[0]
    x_windows = x_windows.reshape(num_windows_B, num_heads * window_size * window_size, head_dim)
    # conv1d: weight shape (num_heads*ws*ws, ws*ws, 1)
    x_windows = torch.nn.functional.conv1d(x_windows, spatial_mlp_weight, spatial_mlp_bias,
                                            groups=num_heads)
    # view back: (num_windows*B, num_heads, ws*ws, head_dim)
    x_windows = x_windows.view(num_windows_B, num_heads, window_size * window_size, head_dim)
    # transpose: (num_windows*B, ws*ws, num_heads, head_dim)
    x_windows = x_windows.transpose(1, 2)
    # reshape: (num_windows*B, ws*ws, C)
    x_windows = x_windows.reshape(num_windows_B, window_size * window_size, C)
    # reshape: (num_windows*B, ws, ws, C)
    x_windows = x_windows.reshape(num_windows_B, window_size, window_size, C)

    # reverse window
    shifted_x = window_reverse(x_windows, window_size, Hp, Wp)

    # reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x

    # remove padding
    if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()

    x = x.view(B, H * W, C)
    x = shortcut + x

    # FFN
    shortcut2 = x
    x = triton_layer_norm(x, norm2_weight, norm2_bias)
    x = torch.nn.functional.linear(x, mlp_fc1_weight, mlp_fc1_bias)
    x = triton_gelu(x)
    # dropout is identity at inference
    x = torch.nn.functional.linear(x, mlp_fc2_weight, mlp_fc2_bias)
    # dropout is identity at inference
    x = shortcut2 + x

    return x


def patch_merging_forward(x, input_resolution, norm_weight, norm_bias, reduction_weight):
    """PatchMerging forward"""
    H, W = input_resolution
    B, L, C = x.shape
    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
    x = x.view(B, -1, 4 * C)

    x = triton_layer_norm(x, norm_weight, norm_bias)
    x = torch.nn.functional.linear(x, reduction_weight)
    return x


def kernel_function(
    p_patch_embed_proj_weight,
    p_patch_embed_proj_bias,
    p_patch_embed_norm_weight,
    p_patch_embed_norm_bias,
    p_layers_0_blocks_0_norm1_weight,
    p_layers_0_blocks_0_norm1_bias,
    p_layers_0_blocks_0_spatial_mlp_weight,
    p_layers_0_blocks_0_spatial_mlp_bias,
    p_layers_0_blocks_0_norm2_weight,
    p_layers_0_blocks_0_norm2_bias,
    p_layers_0_blocks_0_mlp_fc1_weight,
    p_layers_0_blocks_0_mlp_fc1_bias,
    p_layers_0_blocks_0_mlp_fc2_weight,
    p_layers_0_blocks_0_mlp_fc2_bias,
    p_layers_0_blocks_1_norm1_weight,
    p_layers_0_blocks_1_norm1_bias,
    p_layers_0_blocks_1_spatial_mlp_weight,
    p_layers_0_blocks_1_spatial_mlp_bias,
    p_layers_0_blocks_1_norm2_weight,
    p_layers_0_blocks_1_norm2_bias,
    p_layers_0_blocks_1_mlp_fc1_weight,
    p_layers_0_blocks_1_mlp_fc1_bias,
    p_layers_0_blocks_1_mlp_fc2_weight,
    p_layers_0_blocks_1_mlp_fc2_bias,
    p_layers_0_downsample_norm_weight,
    p_layers_0_downsample_norm_bias,
    p_layers_0_downsample_reduction_weight,
    p_layers_1_blocks_0_norm1_weight,
    p_layers_1_blocks_0_norm1_bias,
    p_layers_1_blocks_0_spatial_mlp_weight,
    p_layers_1_blocks_0_spatial_mlp_bias,
    p_layers_1_blocks_0_norm2_weight,
    p_layers_1_blocks_0_norm2_bias,
    p_layers_1_blocks_0_mlp_fc1_weight,
    p_layers_1_blocks_0_mlp_fc1_bias,
    p_layers_1_blocks_0_mlp_fc2_weight,
    p_layers_1_blocks_0_mlp_fc2_bias,
    p_layers_1_blocks_1_norm1_weight,
    p_layers_1_blocks_1_norm1_bias,
    p_layers_1_blocks_1_spatial_mlp_weight,
    p_layers_1_blocks_1_spatial_mlp_bias,
    p_layers_1_blocks_1_norm2_weight,
    p_layers_1_blocks_1_norm2_bias,
    p_layers_1_blocks_1_mlp_fc1_weight,
    p_layers_1_blocks_1_mlp_fc1_bias,
    p_layers_1_blocks_1_mlp_fc2_weight,
    p_layers_1_blocks_1_mlp_fc2_bias,
    p_layers_1_downsample_norm_weight,
    p_layers_1_downsample_norm_bias,
    p_layers_1_downsample_reduction_weight,
    p_layers_2_blocks_0_norm1_weight,
    p_layers_2_blocks_0_norm1_bias,
    p_layers_2_blocks_0_spatial_mlp_weight,
    p_layers_2_blocks_0_spatial_mlp_bias,
    p_layers_2_blocks_0_norm2_weight,
    p_layers_2_blocks_0_norm2_bias,
    p_layers_2_blocks_0_mlp_fc1_weight,
    p_layers_2_blocks_0_mlp_fc1_bias,
    p_layers_2_blocks_0_mlp_fc2_weight,
    p_layers_2_blocks_0_mlp_fc2_bias,
    p_layers_2_blocks_1_norm1_weight,
    p_layers_2_blocks_1_norm1_bias,
    p_layers_2_blocks_1_spatial_mlp_weight,
    p_layers_2_blocks_1_spatial_mlp_bias,
    p_layers_2_blocks_1_norm2_weight,
    p_layers_2_blocks_1_norm2_bias,
    p_layers_2_blocks_1_mlp_fc1_weight,
    p_layers_2_blocks_1_mlp_fc1_bias,
    p_layers_2_blocks_1_mlp_fc2_weight,
    p_layers_2_blocks_1_mlp_fc2_bias,
    p_layers_2_blocks_2_norm1_weight,
    p_layers_2_blocks_2_norm1_bias,
    p_layers_2_blocks_2_spatial_mlp_weight,
    p_layers_2_blocks_2_spatial_mlp_bias,
    p_layers_2_blocks_2_norm2_weight,
    p_layers_2_blocks_2_norm2_bias,
    p_layers_2_blocks_2_mlp_fc1_weight,
    p_layers_2_blocks_2_mlp_fc1_bias,
    p_layers_2_blocks_2_mlp_fc2_weight,
    p_layers_2_blocks_2_mlp_fc2_bias,
    p_layers_2_blocks_3_norm1_weight,
    p_layers_2_blocks_3_norm1_bias,
    p_layers_2_blocks_3_spatial_mlp_weight,
    p_layers_2_blocks_3_spatial_mlp_bias,
    p_layers_2_blocks_3_norm2_weight,
    p_layers_2_blocks_3_norm2_bias,
    p_layers_2_blocks_3_mlp_fc1_weight,
    p_layers_2_blocks_3_mlp_fc1_bias,
    p_layers_2_blocks_3_mlp_fc2_weight,
    p_layers_2_blocks_3_mlp_fc2_bias,
    p_layers_2_blocks_4_norm1_weight,
    p_layers_2_blocks_4_norm1_bias,
    p_layers_2_blocks_4_spatial_mlp_weight,
    p_layers_2_blocks_4_spatial_mlp_bias,
    p_layers_2_blocks_4_norm2_weight,
    p_layers_2_blocks_4_norm2_bias,
    p_layers_2_blocks_4_mlp_fc1_weight,
    p_layers_2_blocks_4_mlp_fc1_bias,
    p_layers_2_blocks_4_mlp_fc2_weight,
    p_layers_2_blocks_4_mlp_fc2_bias,
    p_layers_2_blocks_5_norm1_weight,
    p_layers_2_blocks_5_norm1_bias,
    p_layers_2_blocks_5_spatial_mlp_weight,
    p_layers_2_blocks_5_spatial_mlp_bias,
    p_layers_2_blocks_5_norm2_weight,
    p_layers_2_blocks_5_norm2_bias,
    p_layers_2_blocks_5_mlp_fc1_weight,
    p_layers_2_blocks_5_mlp_fc1_bias,
    p_layers_2_blocks_5_mlp_fc2_weight,
    p_layers_2_blocks_5_mlp_fc2_bias,
    p_layers_2_downsample_norm_weight,
    p_layers_2_downsample_norm_bias,
    p_layers_2_downsample_reduction_weight,
    p_layers_3_blocks_0_norm1_weight,
    p_layers_3_blocks_0_norm1_bias,
    p_layers_3_blocks_0_spatial_mlp_weight,
    p_layers_3_blocks_0_spatial_mlp_bias,
    p_layers_3_blocks_0_norm2_weight,
    p_layers_3_blocks_0_norm2_bias,
    p_layers_3_blocks_0_mlp_fc1_weight,
    p_layers_3_blocks_0_mlp_fc1_bias,
    p_layers_3_blocks_0_mlp_fc2_weight,
    p_layers_3_blocks_0_mlp_fc2_bias,
    p_layers_3_blocks_1_norm1_weight,
    p_layers_3_blocks_1_norm1_bias,
    p_layers_3_blocks_1_spatial_mlp_weight,
    p_layers_3_blocks_1_spatial_mlp_bias,
    p_layers_3_blocks_1_norm2_weight,
    p_layers_3_blocks_1_norm2_bias,
    p_layers_3_blocks_1_mlp_fc1_weight,
    p_layers_3_blocks_1_mlp_fc1_bias,
    p_layers_3_blocks_1_mlp_fc2_weight,
    p_layers_3_blocks_1_mlp_fc2_bias,
    p_norm_weight,
    p_norm_bias,
    p_head_weight,
    p_head_bias,
    x,
):
    # ---- Patch Embedding ----
    # conv2d: (10,3,224,224) -> (10,96,56,56)
    x = torch.nn.functional.conv2d(x, p_patch_embed_proj_weight, p_patch_embed_proj_bias, stride=4)
    # flatten: (10,96,56,56) -> (10,96,3136)
    x = x.flatten(2)
    # transpose: (10,96,3136) -> (10,3136,96)
    x = x.transpose(1, 2)
    # layer_norm
    x = triton_layer_norm(x, p_patch_embed_norm_weight, p_patch_embed_norm_bias)
    # dropout (identity at inference)

    # ---- Layer 0 ----
    # Block 0: shift_size=0, window_size=7, num_heads=3, dim=96, input_resolution=(56,56)
    x = swin_mlp_block_forward(
        x, (56, 56), 3, 7, 0,
        p_layers_0_blocks_0_norm1_weight, p_layers_0_blocks_0_norm1_bias,
        p_layers_0_blocks_0_spatial_mlp_weight, p_layers_0_blocks_0_spatial_mlp_bias,
        p_layers_0_blocks_0_norm2_weight, p_layers_0_blocks_0_norm2_bias,
        p_layers_0_blocks_0_mlp_fc1_weight, p_layers_0_blocks_0_mlp_fc1_bias,
        p_layers_0_blocks_0_mlp_fc2_weight, p_layers_0_blocks_0_mlp_fc2_bias,
    )
    # Block 1: shift_size=3
    x = swin_mlp_block_forward(
        x, (56, 56), 3, 7, 3,
        p_layers_0_blocks_1_norm1_weight, p_layers_0_blocks_1_norm1_bias,
        p_layers_0_blocks_1_spatial_mlp_weight, p_layers_0_blocks_1_spatial_mlp_bias,
        p_layers_0_blocks_1_norm2_weight, p_layers_0_blocks_1_norm2_bias,
        p_layers_0_blocks_1_mlp_fc1_weight, p_layers_0_blocks_1_mlp_fc1_bias,
        p_layers_0_blocks_1_mlp_fc2_weight, p_layers_0_blocks_1_mlp_fc2_bias,
    )
    # Downsample
    x = patch_merging_forward(x, (56, 56),
                               p_layers_0_downsample_norm_weight, p_layers_0_downsample_norm_bias,

    # ---- Layer 1 ----
    # Block 0: shift_size=0, num_heads=6, dim=192, input_resolution=(28,28)
    x = swin_mlp_block_forward(
        x, (28, 28), 6, 7, 0,
        p_layers_1_blocks_0_norm1_weight, p_layers_1_blocks_0_norm1_bias,
        p_layers_1_blocks_0_spatial_mlp_weight, p_layers_1_blocks_0_spatial_mlp_bias,
        p_layers_1_blocks_0_norm2_weight, p_layers_1_blocks_0_norm2_bias,
        p_layers_1_blocks_0_mlp_fc1_weight, p_layers_1_blocks_0_mlp_fc1_bias,
        p_layers_1_blocks_0_mlp_fc2_weight, p_layers_1_blocks_0_mlp_fc2_bias,
    )
    # Block 1: shift_size=3
    x = swin_mlp_block_forward(
        x, (28, 28), 6, 7, 3,
        p_layers_1_blocks_1_norm1_weight, p_layers_1_blocks_1_norm1_bias,
        p_layers_1_blocks_1_spatial_mlp_weight, p_layers_1_blocks_1_spatial_mlp_bias,
        p_layers_1_blocks_1_norm2_weight, p_layers_1_blocks_1_norm2_bias,
        p_layers_1_blocks_1_mlp_fc1_weight, p_layers_1_blocks_1_mlp_fc1_bias,
        p_layers_1_blocks_1_mlp_fc2_weight, p_layers_1_blocks_1_mlp_fc2_bias,
    )
    # Downsample
    x = patch_merging_forward(x, (28, 28),
                               p_layers_1_downsample_norm_weight, p_layers_1_downsample_norm_bias,

    # ---- Layer 2 ----
    # 6 blocks, num_heads=12, dim=384, input_resolution=(14,14)
    shifts = [0, 3, 0, 3, 0, 3]
    blocks_2 = [
        (p_layers_2_blocks_0_norm1_weight, p_layers_2_blocks_0_norm1_bias,