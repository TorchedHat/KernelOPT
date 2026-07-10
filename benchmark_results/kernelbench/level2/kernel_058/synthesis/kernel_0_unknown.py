import torch
import triton
import triton.language as tl
import math

# ConvTranspose3d: weight shape [in_channels, out_channels, kD, kH, kW]
# Input: [N, Ci, D, H, W], Output: [N, Co, OD, OH, OW]
# OD = (D-1)*stride - 2*pad + kD = (16-1)*2 - 2*1 + 3 = 31
# OH = (H-1)*stride - 2*pad + kH = (32-1)*2 - 2*1 + 3 = 63
# OW = (W-1)*stride - 2*pad + kW = 63

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(...):