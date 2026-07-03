"""Helion Tutorial Problem 2: Outer Vector Add.

Source: https://helionlang.com/helion_tutorials.html#problem-2-outer-vector-add
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def broadcast_add_spec(x: Tensor, y: Tensor) -> Tensor:
    return x[None, :] + y[:, None]


@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def broadcast_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = x.new_empty(n1, n0)

    for tile_i, tile_j in hl.tile([n1, n0]):
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        out[tile_i, tile_j] = y_tile[:, None] + x_tile[None, :]

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return broadcast_add_spec(*inputs)

def kernel_function(*inputs):
    return broadcast_add_kernel(*inputs)

def get_inputs():
    return [torch.randn(1142, device="cuda"), torch.randn(512, device="cuda")]


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
