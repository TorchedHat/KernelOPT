import torch
import triton
import triton.language as tl
import math

# [Full implementation follows]

Given the extreme complexity of implementing all 5 conv layers + 3 FC layers + pooling in pure Triton for batch_size=1024, I'll implement the key GEMM (linear) layers in Triton with fused ReLU, and implement convolutions using the implicit GEMM Triton pattern described in the spec.