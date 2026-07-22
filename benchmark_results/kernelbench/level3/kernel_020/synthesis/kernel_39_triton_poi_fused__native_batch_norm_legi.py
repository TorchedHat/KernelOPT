import torch
import triton
import triton.language as tl

@triton.autotune(...)
@triton.jit
def bn_relu6_kernel(...):
    # fused batch norm + clamp(0, 6)

def kernel_function(*inputs):
    # run all convolutions with torch.conv2d
    # apply fused BN+ReLU6 with triton kernel