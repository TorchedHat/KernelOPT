import torch
import triton
import triton.language as tl
import math

# Due to the extreme complexity of implementing all 30+ convolutions as separate Triton kernels,
# I'll implement the key computational kernels and use them throughout.



import torch
import triton
import triton.language as tl