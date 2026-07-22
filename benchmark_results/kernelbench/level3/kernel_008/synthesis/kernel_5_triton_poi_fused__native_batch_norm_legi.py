import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice




The BN here uses running stats (inference mode), so it's just: `out = (x - mean) / sqrt(var + eps) * weight + bias` then ReLU.



import torch
import triton
import triton.language as tl