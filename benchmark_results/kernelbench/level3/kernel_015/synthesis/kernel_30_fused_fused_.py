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
    
    # Compute channel index (NCHW layout: channel = (offset / spatial) % C)
    # numel = N * C * H * W, spatial = H * W = numel / (N * C)
    # channel = (offset // spatial_size) % num_channels
    # But we need spatial_size = numel // (N * num_channels)
    # Actually: in NCHW, index = n * C*H*W + c * H*W + h*W + w
    # So channel = (offset // (H*W)) % C
    # We don't know H*W directly, but numel = N*C*H*W
    # spatial = numel // num_channels // N... we don't have N
    # Better: channel = (offset // spatial_per_channel) % num_channels
    # where spatial_per_channel = numel // num_channels // batch... 
    # Actually simplest: channel index in NCHW is (offset // hw_size) % C
    # hw_size = H*W, total = N*C*H*W
    # offset = n*(C*H*W) + c*(H*W) + h*W + w
    # (offset // hw_size) = n*C + c => % C = c
    # hw_size = numel // num_channels // N... we need N
    # Let's pass hw_size directly
    pass