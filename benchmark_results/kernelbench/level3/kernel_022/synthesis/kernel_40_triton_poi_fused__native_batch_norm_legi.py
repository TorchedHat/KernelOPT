import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + Activation kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_channels', 'numel'],
)
@triton.jit
def fused_bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel,
    eps,
    act_type,  # 0=identity, 1=relu, 2=hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # channel index: numel = N * C * H * W, channel = (offset // (H*W)) % C
    # But we pass spatial_size = H*W separately
    # Actually encode channel from offset
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Determine channel index
    # numel = N * C * spatial, channel_stride = spatial
    # We need channel = (offset // spatial) % C
    # Pass spatial as a parameter
    # Actually let's use a different approach - pass channel stride
    # offset layout: [N, C, H, W] contiguous => channel = (offset // (H*W)) % C
    # We'll pass spatial_size = H*W
    # channel_idx = (offsets // spatial_size) % n_channels
    # But spatial_size is not a constexpr... let's pass it

    # Load per-channel params
    # channel_idx computed from offsets
    # This requires knowing spatial_size at runtime
    # We'll pass it as a parameter
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_channels', 'spatial_size'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = n_channels * spatial_size  # per-batch, but we handle all batches
    # Actually total numel passed separately
    # Let's restructure

    mask = offsets < (n_channels * spatial_size)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // spatial_size) % n_channels

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)