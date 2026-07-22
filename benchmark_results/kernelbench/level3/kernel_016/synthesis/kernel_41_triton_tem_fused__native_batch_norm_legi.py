import torch
import triton
import triton.language as tl
import math

# [Full implementation with conv, bn+relu, pooling kernels]