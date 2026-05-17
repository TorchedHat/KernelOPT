"""Helion Tutorial Problem 1: Constant Add — add a constant to a vector."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[32]))
def kernel_function(x: torch.Tensor) -> torch.Tensor:
    TILE_RANGE = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(TILE_RANGE):
        x_tile = x[tile_n]
        out[tile_n] = x_tile + 10.0
    return out

def get_inputs():
    return [torch.randn(8192, device="cuda")]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    expected = inputs[0] + 10.0
    torch.testing.assert_close(result, expected)
    if "--profile-only" not in sys.argv:
        print("PASS")
