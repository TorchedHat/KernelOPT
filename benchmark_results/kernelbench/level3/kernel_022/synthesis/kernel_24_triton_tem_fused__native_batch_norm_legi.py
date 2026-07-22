import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + activation Triton kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_act_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, num_channels,
    eps,
    act_type: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh6
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # channel index: layout is (N, C, H, W) -> channel = (offset / (H*W)) % C
    # But we pass channel stride info differently
    # We'll compute channel from offset assuming NCHW with spatial = numel // num_channels // batch
    # Actually pass channel_stride = H*W
    chan = (offsets // 1) % num_channels  # placeholder, need channel_stride
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    v = tl.load(var_ptr + chan, mask=mask, other=0.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    x_norm = (x - m) / tl.sqrt(v + eps)
    out = x_norm * w + b
    
    if act_type == 1:
        out = tl.maximum(out, 0.0)
    elif act_type == 2:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)