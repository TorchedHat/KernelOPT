"""Helion Tutorial Problem 6: Long Softmax — softmax of a batch of logits.

Source: https://helionlang.com/helion_tutorials.html#problem-6-long-softmax
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def softmax_spec(x: Tensor) -> Tensor:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@helion.kernel(config=helion.Config(block_sizes=[4, 64, 64, 64]))
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    batch, seq_len = x.size()
    out = torch.empty_like(x)

    for tile_batch in hl.tile(batch):
        max_i = hl.full([tile_batch], float("-inf"), dtype=torch.float32)
        for tile_seq in hl.tile(seq_len):
            x_tile = x[tile_batch, tile_seq]
            max_i = torch.maximum(max_i, torch.amax(x_tile, dim=1))

        denom = hl.zeros([tile_batch], dtype=torch.float32)
        for tile_seq in hl.tile(seq_len):
            x_tile = x[tile_batch, tile_seq]
            denom += torch.exp(x_tile - max_i[:, None]).sum(dim=1)

        for tile_seq in hl.tile(seq_len):
            x_tile = x[tile_batch, tile_seq]
            out[tile_batch, tile_seq] = (
                torch.exp(x_tile - max_i[:, None]) / denom[:, None]
            )

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return softmax_spec(*inputs)

def kernel_function(*inputs):
    return softmax_kernel(*inputs)

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
