import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Y axis: group-instance = batch_idx * num_groups + group_idx
    # 16 batches * 8 groups = 128 group-instances
    group_instance = tl.program_id(1)  # 0..127
    batch_idx = group_instance // 8    # 0..15
    group_idx = group_instance % 8     # 0..7

    # X axis: position within group-instance (0..628863)
    spatial_numel = 628864  # 16 channels * 39304 spatial per group-instance
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < spatial_numel

    # Load group-level scalars ONCE per CTA using first-lane mask
    # in_ptr1/in_ptr2 shape=(16,8,1,1), stride=(8,1,...) -> index = batch_idx*8 + group_idx
    # Use xmask[0] equivalent: load with a single-element block to satisfy mask consistency
    scalar_idx = batch_idx * 8 + group_idx
    mean = tl.load(in_ptr1 + tl.broadcast_to(scalar_idx, [XBLOCK]), xmask, eviction_policy='evict_last')
    var  = tl.load(in_ptr2 + tl.broadcast_to(scalar_idx, [XBLOCK]), xmask, eviction_policy='evict_last')
    # Precompute inv_std once per CTA
    inv_std = libdevice.rsqrt(var * 1.5908420085906982e-06 + 1e-05)

    # Decode channel within group (0..15) for weight/bias lookup
    chan_in_group = xindex // 39304   # 0..15
    channel_idx = group_idx * 16 + chan_in_group  # global channel 0..127

    # Load weight/bias (16 unique values per CTA, broadcast per channel stripe)
    weight = tl.load(in_ptr3 + channel_idx, xmask, eviction_policy='evict_last')
    bias   = tl.load(in_ptr4 + channel_idx, xmask, eviction_policy='evict_last')

    # Compute flat offset into in_ptr0/out_ptr0
    # in_ptr0 shape=(16,128,34,34,34), stride=(5030912,39304,1156,34,1)
    spatial_offset = (batch_idx * 5030912 +
                      group_idx * 16 * 39304 +
                      xindex)

    # Load spatial data (coalesced: contiguous in memory along xindex)
    x_val = tl.load(in_ptr0 + spatial_offset, xmask)

    # Compute: ReLU -> normalize -> affine
    x_relu = tl.maximum(x_val, 0.0)
    x_norm = (x_relu - mean) * inv_std
    out    = x_norm * weight + bias

    tl.store(out_ptr0 + spatial_offset, out, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 80494592
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_native_group_norm_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 80494592)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")
