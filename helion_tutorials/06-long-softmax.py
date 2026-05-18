"""Helion Tutorial Problem 6: Long Softmax — softmax of a batch of logits."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[4, 64, 64, 64]))
def kernel_function(x: torch.Tensor) -> torch.Tensor:
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
            out[tile_batch, tile_seq] = torch.exp(x_tile - max_i[:, None]) / denom[:, None]
    return out

def get_inputs():
    return [torch.randn(4, 200, device="cuda")]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    x = inputs[0]
    x_max = x.max(1, keepdim=True)[0]
    x_exp = (x - x_max).exp()
    expected = x_exp / x_exp.sum(1, keepdim=True)
    torch.testing.assert_close(result, expected)
    if "--profile-only" not in sys.argv:
        print("PASS")
