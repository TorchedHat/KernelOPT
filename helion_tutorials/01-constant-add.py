"""Helion Tutorial Problem 1: Constant Add — add a constant to a vector.

Source: https://helionlang.com/helion_tutorials.html#problem-1-constant-add
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


# PyTorch reference (from website)
def add_spec(x: Tensor) -> Tensor:
    return x + 10.


# Helion kernel (from website)
@helion.kernel(config=helion.Config(block_sizes=[32]))
def add_kernel(x: torch.Tensor) -> torch.Tensor:
    TILE_RANGE = x.size()
    out = torch.empty_like(x)

    for tile_n in hl.tile(TILE_RANGE):
        x_tile = x[tile_n]
        out[tile_n] = x_tile + 10.0

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return add_spec(*inputs)

def kernel_function(*inputs):
    return add_kernel(*inputs)

def get_inputs():
    return [torch.randn(8192, device="cuda")]


if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    if "--profile-only" in sys.argv:
        kernel_function(*inputs)
        torch.cuda.synchronize()
    else:
        kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
