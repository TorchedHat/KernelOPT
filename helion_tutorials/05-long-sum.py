"""Helion Tutorial Problem 5: Long Sum — sum of a batch of numbers."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[4, 64]))
def kernel_function(x: torch.Tensor) -> torch.Tensor:
    batch, seq_len = x.size()
    out = torch.empty(batch, dtype=x.dtype, device=x.device)
    for tile_batch in hl.tile(batch):
        acc = hl.zeros([tile_batch], dtype=torch.float32)
        for tile_seq in hl.tile(seq_len):
            chunk = x[tile_batch, tile_seq]
            acc += torch.sum(chunk, dim=1)
        out[tile_batch] = acc
    return out

def get_inputs():
    return [torch.randn(4, 200, device="cuda")]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    expected = inputs[0].sum(1)
    torch.testing.assert_close(result, expected)
    if "--profile-only" not in sys.argv:
        print("PASS")
