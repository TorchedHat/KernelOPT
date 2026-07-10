import torch
import triton
import triton.language as tl
import math

# ... kernels ...

def get_inputs():
    p_conv_transpose_weight = torch.rand(3, 16, 3, 3, 3)
    p_conv_transpose_bias = torch.rand(16)
    p_group_norm_weight = torch.rand(16)
    p_group_norm_bias = torch.rand(16)
    x = torch.rand(128, 3, 16, 32, 32)
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_group_norm_weight, p_group_norm_bias, x]