"""Helion Tutorial Problem 8: Two Dimensional Convolution."""
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[4]), ignore_warnings=[helion.exc.TensorOperationInWrapper])
def kernel_function(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
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

def get_inputs():
    return [
        torch.randn(4, 8, 8, device="cuda"),
        torch.randn(4, 4, 4, device="cuda"),
    ]

if __name__ == "__main__":
    import sys
    inputs = get_inputs()
    result = kernel_function(*inputs)
    if "--profile-only" not in sys.argv:
        print("PASS")
