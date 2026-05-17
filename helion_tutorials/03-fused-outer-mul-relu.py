"""Helion Tutorial Problem 3: Fused Outer Multiplication with ReLU."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def kernel_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)
    for tile_i, tile_j in hl.tile([n1, n0]):
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        out[tile_i, tile_j] = torch.relu(y_tile[:, None] * x_tile[None, :])
    return out

def get_inputs():
    return [torch.randn(512, device="cuda"), torch.randn(512, device="cuda")]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    expected = torch.relu(inputs[0][None, :] * inputs[1][:, None])
    torch.testing.assert_close(result, expected)
    if "--profile-only" not in sys.argv:
        print("PASS")
