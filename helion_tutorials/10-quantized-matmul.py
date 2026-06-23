"""Helion Tutorial Problem 10: Quantized Matrix Multiplication (4-bit packed).

Source: https://helionlang.com/helion_tutorials.html#problem-10-quantized-matrix-multiplication
"""
import torch
import helion
import helion.language as hl
from torch import Tensor

FPINT = 32 // 4
GROUP = 8


def extract_4bit(x: torch.Tensor) -> torch.Tensor:
    """Extract 8 x 4-bit values from packed int32 tensor."""
    over = torch.arange(8, device=x.device) * 4
    mask = 2**4 - 1
    return (x[..., None] >> over) & mask


def quant_dot_spec(scale: Tensor, offset: Tensor,
                   weight: Tensor, activation: Tensor) -> Tensor:
    offset = offset.view(32, 1)
    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = extract_4bit(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    return (scale * (extract_4bit(weight).view(-1, 64) - offset)) @ activation


@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def quant_dot_kernel(
    scale: torch.Tensor, offset: torch.Tensor,
    weight: torch.Tensor, activation: torch.Tensor,
) -> torch.Tensor:
    n_out, n_groups = scale.size()
    mid, n_in = activation.size()
    out = torch.empty([n_out, n_in], dtype=scale.dtype, device=scale.device)

    shifts = torch.arange(FPINT, device=scale.device) * 4
    mask = 2**4 - 1

    for tile_out, tile_in in hl.tile([n_out, n_in]):
        acc = hl.zeros([tile_out, tile_in], dtype=torch.float32)

        for group_idx in range(n_groups):
            scale_group = scale[tile_out, group_idx]

            w_packed = weight[tile_out, group_idx]
            w_nibbles = (w_packed[:, None] >> shifts[None, :]) & mask
            w_nibbles = w_nibbles.to(scale.dtype)

            o_packed = offset[tile_out]
            o_nibble = ((o_packed >> (group_idx * 4)) & mask).to(scale.dtype)

            dequant = scale_group[:, None] * (w_nibbles - o_nibble[:, None])

            act_group = activation[group_idx * 8 : (group_idx + 1) * 8, tile_in]

            acc = acc + torch.matmul(dequant, act_group)

        out[tile_out, tile_in] = acc

    return out


# ===== kpt interface =====

def reference_function(*inputs):
    return quant_dot_spec(*inputs)

def kernel_function(*inputs):
    return quant_dot_kernel(*inputs)

def get_inputs():
    return [
        torch.randn(32, 8, device="cuda"),
        torch.randint(-2**31, 2**31, (32,), device="cuda", dtype=torch.int32),
        torch.randint(-2**31, 2**31, (32, 8), device="cuda", dtype=torch.int32),
        torch.randn(64, 32, device="cuda"),
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
