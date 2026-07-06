import torch
import triton
import triton.language as tl

# ConvTranspose2d: output[n,co,oh,ow] = sum over ci,kh,kw of input[n,ci,ih,iw] * weight[ci,co,kh,kw]
# where ih = (oh + padding - kh*dilation) / stride (must be integer and in bounds)
#       iw = (ow + padding - kw*dilation) / stride (must be integer and in bounds)