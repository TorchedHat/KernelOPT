import torch
import triton
import triton.language as tl
import torch.nn.functional as F

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
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Channel index: layout is (N, C, H, W) so channel = (offset / (H*W)) % C
    # But we need channel from flat index
    # numel = N * C * H * W, channel stride = H*W
    # channel_idx = (offset // spatial_size) % num_channels
    # We pass num_channels and compute spatial_size = numel // (N * C) ... 
    # Actually easier: channel = (offset // (numel // num_channels // (numel // num_channels // 1))) 
    # Let's just use: channel_idx = (offsets // (numel // num_channels)) % num_channels
    # Wait, numel = N*C*H*W, numel//num_channels = N*H*W... no that's wrong
    # Actually spatial_per_channel = numel // num_channels (= N*H*W)
    # channel_idx = (offsets // (numel // num_channels)) % num_channels ... no
    # offsets goes 0..N*C*H*W-1
    # In NCHW: offset = n*(C*H*W) + c*(H*W) + h*W + w
    # channel = (offset // (H*W)) % C = (offset // spatial) % C
    # spatial = numel / (N * C) ... we need N
    # Let's pass spatial_size directly
    
    channel_idx = (offsets // (numel // num_channels)) % num_channels
    # Hmm this is wrong. Let me reconsider.