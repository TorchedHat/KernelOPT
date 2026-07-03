import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'BLOCK_R': 64}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 1, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'BLOCK_R': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1, 'BLOCK_R': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'BLOCK_R': 128}, num_warps=8, num_stages=1),
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
    # Each program handles XBLOCK=1 spatial positions in channels-last layout
    # Features for each spatial position are contiguous (stride-1) in memory
    pid = tl.program_id(0)
    x_start = pid * XBLOCK
    x_offsets = x_start + tl.arange(0, XBLOCK)
    x_mask = x_offsets < xnumel

    r_offsets = tl.arange(0, BLOCK_R)  # [BLOCK_R]
    r_mask = r_offsets < rnumel        # [BLOCK_R]

    # In channels-last layout [B, H, W, C], features are stride-1:
    # index = spatial_pos * rnumel + feat
    # = x_offsets[:, None] * rnumel + r_offsets[None, :]
    # This is FULLY COALESCED: adjacent threads access adjacent memory
    indices = x_offsets[:, None] * rnumel + r_offsets[None, :]  # [XBLOCK, BLOCK_R]
    mask_2d = x_mask[:, None] & r_mask[None, :]                 # [XBLOCK, BLOCK_R]

    vals = tl.load(in_ptr + indices, mask=mask_2d, other=0.0)   # [XBLOCK, BLOCK_R]
    vals_f32 = vals.to(tl.float32)

    sq = vals_f32 * vals_f32                # [XBLOCK, BLOCK_R]
    sum_sq = tl.sum(sq, axis=1)            # [XBLOCK]

    mean_sq = sum_sq / rnumel
    rms = tl.sqrt(mean_sq + eps)           # [XBLOCK]

    # Normalize and store
    normalized = vals_f32 / rms[:, None]   # [XBLOCK, BLOCK_R]

    tl.store(out_ptr + indices, normalized.to(tl.float32), mask=mask_2d)

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