"""Helion Tutorial Problem 3: Fused Outer Multiplication with ReLU.

Source: https://helionlang.com/helion_tutorials.html#problem-3-fused-outer-multiplication
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def mul_relu_block_spec(x: Tensor, y: Tensor) -> Tensor:
    return torch.relu(x[None, :] * y[:, None])


@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def mul_relu_block_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)

    for tile_i, tile_j in hl.tile([n1, n0]):
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        out[tile_i, tile_j] = torch.relu(y_tile[:, None] * x_tile[None, :])

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return mul_relu_block_spec(*inputs)

def kernel_function(*inputs):
    return mul_relu_block_kernel(*inputs)

def get_inputs():
    return [torch.randn(512, device="cuda"), torch.randn(512, device="cuda")]


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
