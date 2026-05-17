"""Helion Tutorial Problem 9: Batched Matrix Multiplication."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[4, [32, 32], 32]))
def kernel_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

def get_inputs():
    return [
        torch.randn(4, 32, 32, device="cuda", dtype=torch.float16),
        torch.randn(4, 32, 32, device="cuda", dtype=torch.float16),
    ]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    expected = inputs[0] @ inputs[1]
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    if "--profile-only" not in sys.argv:
        print("PASS")
