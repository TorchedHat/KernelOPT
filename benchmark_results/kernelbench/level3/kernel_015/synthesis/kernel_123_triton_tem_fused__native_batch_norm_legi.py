import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_channels', 'spatial'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    # Layout: (N, C, H, W) -> flat index, channel = (idx // spatial) % n_channels
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n_channels * spatial  # per-batch total, but we flatten everything
    # Actually total elements = batch * n_channels * spatial
    # We need to figure out channel from flat index
    # flat index: batch * n_channels * spatial + c * spatial + hw
    # channel = (flat_idx // spatial) % n_channels
    
    mask = offsets < (tl.num_programs(0) * BLOCK_SIZE)  # will be masked properly
    # We need total size passed in
    # Let's use n_channels * spatial as "stride" info
    # flat_idx -> channel = (flat_idx // spatial) % n_channels
    
    c = (offsets // spatial) % n_channels
    
    x = tl.load(x_ptr + offsets, mask=offsets < tl.num_programs(0) * BLOCK_SIZE, other=0.0)
    mean = tl.load(mean_ptr + c, mask=offsets < tl.num_programs(0) * BLOCK_SIZE, other=0.0)
    var = tl.load(var_ptr + c, mask=offsets < tl.num_programs(0) * BLOCK_SIZE, other=1.0)
    w = tl.load(weight_ptr + c, mask=offsets < tl.num_programs(0) * BLOCK_SIZE, other=1.0)
    b = tl.load(bias_ptr + c, mask=offsets < tl.num_programs(0) * BLOCK_SIZE, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=offsets < tl.num_programs(0) * BLOCK_SIZE)