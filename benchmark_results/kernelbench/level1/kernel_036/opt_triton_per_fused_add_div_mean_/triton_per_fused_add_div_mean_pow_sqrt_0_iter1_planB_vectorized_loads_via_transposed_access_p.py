import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64, 'BLOCK_R': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64, 'BLOCK_R': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'BLOCK_R': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'BLOCK_R': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'BLOCK_R': 256}, num_warps=8, num_stages=2),
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
    # Each program handles BLOCK_R spatial positions (stride-1, coalesced)
    # XBLOCK covers the feature (C) dimension (stride-262144, reduction axis)
    pid = tl.program_id(0)
    sp_start = pid * BLOCK_R
    sp_offsets = sp_start + tl.arange(0, BLOCK_R)  # spatial positions, stride=1
    r_offsets = tl.arange(0, XBLOCK)               # feature indices

    sp_mask = sp_offsets < xnumel
    r_mask = r_offsets < rnumel
    mask_2d = r_mask[:, None] & sp_mask[None, :]   # [XBLOCK, BLOCK_R]

    # in_ptr layout: [112, 64, 512, 512] with strides [16777216, 262144, 512, 1]
    # sp_offsets indexes into [112, 512, 512] = 29360128 spatial positions
    # sp_offsets = batch_idx * 262144 + spatial_pos
    spatial_size = 262144  # 512 * 512
    batch_idx = sp_offsets // spatial_size   # [BLOCK_R]
    spatial_pos = sp_offsets % spatial_size  # [BLOCK_R]

    batch_base = batch_idx * 16777216  # batch_idx * (rnumel * spatial_size) [BLOCK_R]

    # Apply vectorization hints on contiguous spatial dimension (stride-1)
    sp_contiguous = tl.max_contiguous(tl.multiple_of(spatial_pos, 16), 16)

    # stride between warp threads = 1 (coalesced!)
    # indices shape: [XBLOCK, BLOCK_R]
    indices = batch_base[None, :] + r_offsets[:, None] * spatial_size + sp_contiguous[None, :]

    vals = tl.load(in_ptr + indices, mask=mask_2d, other=0.0)  # [XBLOCK, BLOCK_R]
    vals_f32 = vals.to(tl.float32)

    sq = vals_f32 * vals_f32  # [XBLOCK, BLOCK_R]
    sum_sq = tl.sum(sq, axis=0)  # reduce over XBLOCK (feature) axis -> [BLOCK_R]

    mean_sq = sum_sq / rnumel
    rms = tl.sqrt(mean_sq + eps)  # [BLOCK_R]

    # Normalize and store
    normalized = vals_f32 / rms[None, :]  # [XBLOCK, BLOCK_R]

    # Store back
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