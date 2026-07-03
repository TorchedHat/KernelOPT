import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'UNROLL_FACTOR': 1}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL_FACTOR': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL_FACTOR': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'UNROLL_FACTOR': 2}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL_FACTOR': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL_FACTOR': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'UNROLL_FACTOR': 4}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL_FACTOR': 4}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL_FACTOR': 4}, num_warps=8, num_stages=1),
    ],
    key=['numel'],
)
@triton.jit
def elu_kernel(
    in_ptr,
    out_ptr,
    numel,
    ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE * UNROLL_FACTOR
    arange = tl.arange(0, BLOCK_SIZE)

    for i in tl.static_range(UNROLL_FACTOR):  # fully unrolled at compile time
        offsets = base + i * BLOCK_SIZE + arange
        mask = offsets < numel
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        # ELU: x if x > 0 else ALPHA * (exp(x) - 1)
        is_positive = x > 0.0
        neg_val = ALPHA * libdevice.expm1(x)
        result = tl.where(is_positive, x, neg_val)
        tl.store(out_ptr + offsets, result, mask=mask)

def kernel_function(*inputs):
    x = inputs[0]
    alpha = 1.0

    output = torch.empty_like(x)
    numel = x.numel()

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    elu_kernel[grid](
        x,
        output,
        numel,
        alpha,
    )

    return output


def get_inputs():
    x = torch.rand(4096, 393216, device='cuda', dtype=torch.float32)
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
        import torch.nn.functional as F
        ref = F.elu(inputs[0], alpha=1.0)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")