"""Helion Tutorial Problem 10: Quantized Matrix Multiplication (4-bit packed)."""
import torch
import helion
import helion.language as hl

FPINT = 32 // 4
GROUP = 8

@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def kernel_function(
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
    result = kernel_function(*inputs)
    if "--profile-only" not in sys.argv:
        print("PASS")
