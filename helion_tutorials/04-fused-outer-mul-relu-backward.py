"""Helion Tutorial Problem 4: Fused Outer Multiplication - Backwards.

Source: https://helionlang.com/helion_tutorials.html#problem-4-fused-outer-multiplication-backwards
"""
import torch
import helion
import helion.language as hl
from torch import Tensor


def mul_relu_block_back_spec(x: Tensor, y: Tensor, dz: Tensor) -> tuple[Tensor, Tensor]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
    return grad_x, grad_y


@helion.kernel(config=helion.Config(block_sizes=[[32, 32], [32, 32]]))
def mul_relu_block_back_kernel(
    x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n0 = x.size(1)
    n1 = x.size(0)
    dx = torch.empty_like(x)
    dy = torch.empty([n1], dtype=x.dtype, device=x.device)

    for tile_i, tile_j in hl.tile([n1, n0]):
        x_tile = x[tile_i, tile_j]
        y_tile = y[tile_i]
        dz_tile = dz[tile_i, tile_j]
        relu_mask = (x_tile * y_tile[:, None]) > 0
        dx[tile_i, tile_j] = dz_tile * relu_mask * y_tile[:, None]

    for tile_i2 in hl.tile(n1):
        acc_dy = hl.zeros([tile_i2], dtype=torch.float32)
        for tile_j2 in hl.tile(n0):
            x2 = x[tile_i2, tile_j2]
            y2 = y[tile_i2]
            dz2 = dz[tile_i2, tile_j2]
            mask2 = (x2 * y2[:, None]) > 0
            acc_dy += torch.sum(dz2 * mask2 * x2, dim=1)
        dy[tile_i2] = acc_dy

    return dx, dy


# ===== kpt interface =====

def reference_function(*inputs):
    return mul_relu_block_back_spec(*inputs)

def kernel_function(*inputs):
    return mul_relu_block_back_kernel(*inputs)

def get_inputs():
    return [
        torch.randn(512, 1024, device="cuda"),
        torch.randn(512, device="cuda"),
        torch.randn(512, 1024, device="cuda"),
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
