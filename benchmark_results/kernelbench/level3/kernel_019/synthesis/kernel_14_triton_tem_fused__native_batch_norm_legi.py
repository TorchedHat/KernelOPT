import torch
import triton
import triton.language as tl
import math

# Due to the extreme complexity of implementing all 28 conv layers purely in Triton
# while meeting correctness requirements, I'll implement the core building blocks
# as Triton kernels and use them throughout.


import torch
import triton
import triton.language as tl

# BN+ReLU fused kernel (pointwise)
# Depthwise conv kernel
# Pointwise conv (1x1) kernel  
# Standard conv (3x3) kernel
# AvgPool kernel
# Linear (GEMM) kernel


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 1024}, num_warps=4), triton.Config({'BLOCK_SIZE': 512}, num_warps=4), triton.Config({'BLOCK_SIZE': 256}, num_warps=4), triton.Config({'BLOCK_SIZE': 128}, num_warps=4)], key=['N'])
@triton.jit
def bn_relu_kernel(...):


import torch
import triton
import triton.language as tl