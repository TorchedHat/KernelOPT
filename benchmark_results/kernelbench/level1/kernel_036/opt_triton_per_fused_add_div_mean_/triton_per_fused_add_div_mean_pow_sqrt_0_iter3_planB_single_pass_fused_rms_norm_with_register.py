import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'BLOCK_R': 64}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'BLOCK_R': 64}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 256, 'BLOCK_R': 64}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 512, 'BLOCK_R': 64}, num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 256, 'BLOCK_R': 64}, num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 512, 'BLOCK_R': 64}, num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 1024,'BLOCK_R': 64}, num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 2048,'BLOCK_R': 64}, num_warps=16, num_stages=3),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def rms_norm_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    rnumel,
    eps,
    XBLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Each program handles XBLOCK spatial positions
    # For each spatial position, we reduce over rnumel (64) features
    pid = tl.program_id(0)
    x_start = pid * XBLOCK
    x_offsets = x_start + tl.arange(0, XBLOCK)
    x_mask = x_offsets < xnumel

    # Compute batch index and spatial position within batch
    spatial_size = 262144  # 512 * 512
    batch_idx = x_offsets // spatial_size  # [XBLOCK]
    spatial_pos = x_offsets % spatial_size  # [XBLOCK]

    batch_base = batch_idx * 16777216  # [XBLOCK]  (16777216 = 64 * 262144)
    sp_base = spatial_pos              # [XBLOCK]

    # Single pass: load once into register cache, accumulate sum-of-squares with FMA
    # Use tl.static_range(BLOCK_R) so the compiler can index the list at compile time
    channels = [tl.zeros([XBLOCK], dtype=tl.float32)] * BLOCK_R
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for r in tl.static_range(BLOCK_R):
        idx = batch_base + r * 262144 + sp_base  # [XBLOCK], coalesced in spatial dim
        v = tl.load(in_ptr + idx, mask=x_mask, other=0.0).to(tl.float32)
        channels[r] = v  # cache in register — no second load needed
        acc = tl.math.fma(v, v, acc)  # FMA: v*v + acc in one instruction

    rms_inv = 1.0 / tl.sqrt(acc / rnumel + eps)  # precompute reciprocal once [XBLOCK]

    # Normalize from register cache — NO memory loads in this loop
    for r in tl.static_range(BLOCK_R):
        idx = batch_base + r * 262144 + sp_base  # [XBLOCK]
        tl.store(out_ptr + idx, channels[r] * rms_inv, mask=x_mask, eviction_policy='evict_first')

def kernel_function(x):
    # x: [112, 64, 512, 512]
    assert x.is_cuda and x.dtype == torch.float32
    B, C, H, W = x.shape
    eps = 1e-5

    out = torch.empty_like(x)

    # xnumel = B * H * W = 112 * 512 * 512 = 29360128
    xnumel = B * H * W
    rnumel = C  # 64

    def grid(meta):
        return (triton.cdiv(xnumel, meta['XBLOCK']),)

    rms_norm_kernel[grid](
        x,
        out,
        xnumel,
        rnumel,
        eps,
    )

    return out


def get_inputs():
    x = torch.rand(112, 64, 512, 512, device='cuda', dtype=torch.float32)
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    x = inputs[0]

    output = kernel_function(x)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-5)
        expected = x / rms
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match PyTorch.")