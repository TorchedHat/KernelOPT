"""Helion Tutorial Problem 2: Outer Vector Add."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def kernel_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = x.new_empty(n1, n0)
    for tile_i, tile_j in hl.tile([n1, n0]):
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        out[tile_i, tile_j] = y_tile[:, None] + x_tile[None, :]
    return out

def get_inputs():
    return [torch.randn(1142, device="cuda"), torch.randn(512, device="cuda")]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    expected = inputs[0][None, :] + inputs[1][:, None]
    torch.testing.assert_close(result, expected)
    if "--profile-only" not in sys.argv:
        print("PASS")
