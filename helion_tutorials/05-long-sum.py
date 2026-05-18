"""Helion Tutorial Problem 5: Long Sum — sum of a batch of numbers.

Source: https://helionlang.com/helion_tutorials.html#problem-5-long-sum
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def sum_spec(x: Tensor) -> Tensor:
    return x.sum(1)


@helion.kernel(config=helion.Config(block_sizes=[4, 64]))
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    batch, seq_len = x.size()
    out = torch.empty(batch, dtype=x.dtype, device=x.device)

    for tile_batch in hl.tile(batch):
        acc = hl.zeros([tile_batch], dtype=torch.float32)
        for tile_seq in hl.tile(seq_len):
            chunk = x[tile_batch, tile_seq]
            acc += torch.sum(chunk, dim=1)
        out[tile_batch] = acc

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return sum_spec(*inputs)

def kernel_function(*inputs):
    return sum_kernel(*inputs)

def get_inputs():
    return [torch.randn(4, 200, device="cuda")]


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
