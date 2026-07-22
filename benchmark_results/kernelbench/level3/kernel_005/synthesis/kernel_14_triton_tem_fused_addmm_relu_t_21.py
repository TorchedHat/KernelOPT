import torch
import triton
import triton.language as tl


After careful analysis, the numerical errors come from using `allow_tf32=True` or insufficient accumulation precision. I'll use `allow_tf32=False` and ensure float32 accumulators everywhere.