import torch
import triton
import triton.language as tl

# ConvTranspose2d: input (N, Ci, H, W), weight (Ci, Co, KH, KW)
# Output (N, Co, OH, OW) where OH = (H-1)*stride - 2*pad + KH + output_padding
# For each output (n, co, oh, ow):
#   out[n,co,oh,ow] = sum over ci,kh,kw of: x[n,ci,ih,iw] * w[ci,co,kh,kw]
#   where ih = (oh + pad - kh) / stride (only if divisible)