import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        # num_stages=1 (baseline, no pipelining)
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=1),
        # num_stages=2 (1 prefetch buffer ahead)
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        # num_stages=4 (3 prefetch buffers ahead)
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
        # large block configs for streaming workloads
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=1),
    ],
    key=['numel'],
)
@triton.jit
def selu_kernel(
    in_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # SELU constants
    # Pre-fold SCALE * ALPHA into one constant to enable FMA fusion
    SCALE: tl.constexpr       = 1.0507009873554804934193349852946
    SCALE_ALPHA: tl.constexpr = 1.7580993408473766  # = SCALE * ALPHA exactly

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')

    # SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
    is_positive = x > 0.0
    pos_val = x * SCALE
    # expm1(x) = exp(x) - 1, numerically stable; single multiply, FMA-eligible
    neg_val = SCALE_ALPHA * libdevice.expm1(x)
    result = tl.where(is_positive, pos_val, neg_val)

    tl.store(out_ptr + offsets, result, mask=mask, cache_modifier='.cs')

def kernel_function(*inputs):
    x, = inputs
    output = torch.empty_like(x)
    numel = x.numel()

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    selu_kernel[grid](x, output, numel)

    return output


def get_inputs():
    x = torch.rand(4096, 393216, dtype=torch.float32, device='cuda')
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        x = inputs[0]
        ref = torch.selu(x)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Kernel output differs from reference! max_diff={max_diff}"
        print("Kernel executed successfully and matches reference.")