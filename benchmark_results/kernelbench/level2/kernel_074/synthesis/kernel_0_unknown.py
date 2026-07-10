import torch
import triton
import triton.language as tl

# ConvTranspose3d kernel + fused epilogue
# Then MaxPool3d kernel