import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index = offset % n_channels (for NCHW layout where channel varies fastest in our flattened view)
    # Actually for NCHW: index = n*C*H*W + c*H*W + h*W + w, so c = (index // (H*W)) % C
    # But we need to figure out channel from flat index
    # For bias add: bias[c] where c = (offset / spatial_size) % C
    # We'll pass n_channels and compute channel index
    chan = (offsets // 1) % n_channels  # placeholder - need spatial size
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    result = x + bias
    result = tl.maximum(result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)