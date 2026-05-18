"""Helion Tutorial Problem 9: Batched Matrix Multiplication.

Source: https://helionlang.com/helion_tutorials.html#problem-9-matrix-multiplication
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def dot_spec(x: Tensor, y: Tensor) -> Tensor:
    return x @ y


@helion.kernel(config=helion.Config(block_sizes=[4, [32, 32], 32]))
def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch, m, k = x.size()
    _, k, n = y.size()

    out = torch.empty([batch, m, n], dtype=x.dtype, device=x.device)

    for tile_batch in hl.tile(batch):
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_batch, tile_m, tile_n], dtype=torch.float32)

            for tile_k in hl.tile(k):
                acc = torch.baddbmm(acc, x[tile_batch, tile_m, tile_k], y[tile_batch, tile_k, tile_n])

            out[tile_batch, tile_m, tile_n] = acc

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return dot_spec(*inputs)

def kernel_function(*inputs):
    return dot_kernel(*inputs)

def get_inputs():
    return [
        torch.randn(4, 32, 32, device="cuda", dtype=torch.float16),
        torch.randn(4, 32, 32, device="cuda", dtype=torch.float16),
    ]


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
