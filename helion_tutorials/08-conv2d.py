"""Helion Tutorial Problem 8: Two Dimensional Convolution.

Source: https://helionlang.com/helion_tutorials.html#problem-8-two-dimensional-convolution
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def conv2d_spec(x: Tensor, k: Tensor) -> Tensor:
    z = torch.zeros(4, 8, 8, device=x.device)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
    return z


@helion.kernel(config=helion.Config(block_sizes=[4]), ignore_warnings=[helion.exc.TensorOperationInWrapper])
def conv2d_kernel(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    batch, h, w = x.size()
    kh, kw = k.size()[1:]

    out = torch.empty_like(x)

    x_padded = torch.nn.functional.pad(x, (0, kw, 0, kh, 0, 0), value=0.0)

    for tile_batch in hl.tile(batch):
        for i in range(h):
            for j in range(w):
                patch = x_padded[tile_batch, i:i+kh, j:j+kw]
                out[tile_batch, i, j] = (k[tile_batch,:,:] * patch).sum(1).sum(1)

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return conv2d_spec(*inputs)

def kernel_function(*inputs):
    return conv2d_kernel(*inputs)

def get_inputs():
    return [
        torch.randn(4, 8, 8, device="cuda"),
        torch.randn(4, 4, 4, device="cuda"),
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
