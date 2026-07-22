import torch
import triton
import triton.language as tl

1. For each layer and timestep (sequential), compute ih = x @ W_ih.T + b_ih and hh = h @ W_hh.T + b_hh using tl.dot