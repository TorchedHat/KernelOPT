import torch
import triton
import triton.language as tl

# Kernel 1: Conv3d as implicit GEMM
# Kernel 2: Fused MaxPool3d + LogSumExp + ReLU (with bias already added)