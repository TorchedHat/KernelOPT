import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'BLOCK_R': 64}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 128, 'BLOCK_R': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'BLOCK_R': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'BLOCK_R': 64}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256, 'BLOCK_R': 64}, num_warps=8, num_stages=1),
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

    batch_base = batch_idx * 16777216  # [XBLOCK]

    # Build 2D index [XBLOCK, BLOCK_R] covering all channels
    r_offsets = tl.arange(0, BLOCK_R)  # [BLOCK_R], BLOCK_R=64=rnumel
    idx_2d = (batch_base[:, None]
              + r_offsets[None, :] * 262144
              + spatial_pos[:, None])   # [XBLOCK, BLOCK_R]
    mask_2d = x_mask[:, None] & (r_offsets[None, :] < rnumel)

    # Single load: all channels into register tile -- no second DRAM pass
    vals = tl.load(in_ptr + idx_2d, mask=mask_2d, other=0.0).to(tl.float32)  # [XBLOCK, BLOCK_R]

    # Compute RMS over channel axis
    sum_sq = tl.sum(vals * vals, axis=1)          # [XBLOCK]
    rms_inv = tl.rsqrt(sum_sq / rnumel + eps)     # [XBLOCK], single MUFU.RSQ instruction

    # Normalize in-register and store -- no reload from DRAM
    normalized = vals * rms_inv[:, None]           # [XBLOCK, BLOCK_R]
    tl.store(out_ptr + idx_2d, normalized.to(tl.float32), mask=mask_2d)

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