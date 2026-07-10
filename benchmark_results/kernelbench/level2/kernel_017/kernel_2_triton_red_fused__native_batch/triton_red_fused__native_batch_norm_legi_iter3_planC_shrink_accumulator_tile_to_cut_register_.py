import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 32},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 32},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 16},  num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_convolution_div_view_2(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 15876
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    # xindex shape [1, XBLOCK]: channel varies across threads (fast axis = coalesced)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    batch_id = xindex // 128   # shape [1, XBLOCK]
    chan_id  = xindex % 128    # shape [1, XBLOCK], contiguous 0..127

    # r0_base shape [R0_BLOCK, 1]: spatial varies down rows
    r0_base = tl.arange(0, R0_BLOCK)[:, None]

    # Bias load: contiguous across chan_id -- COALESCED
    tmp1 = tl.load(in_ptr1 + chan_id, xmask, eviction_policy='evict_last', other=0.0)

    tmp4_mean = tl.zeros([R0_BLOCK, XBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([R0_BLOCK, XBLOCK], tl.float32)
    tmp4_weight = tl.zeros([R0_BLOCK, XBLOCK], tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base  # shape [R0_BLOCK, 1]
        r0_mask = r0_index < r0_numel   # shape [R0_BLOCK, 1]
        roffset = r0_offset
        r0_1 = r0_index
        # Load: chan_id contiguous across warp threads -- COALESCED 128-wide
        tmp0 = tl.load(in_ptr0 + (128*r0_1 + 2032128*batch_id + chan_id), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK, XBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)

    # Welford final reduction over R0_BLOCK dim (axis=0)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 0)
    tmp4 = tmp5[None, :]  # shape [1, XBLOCK]
    tmp8 = tmp6[None, :]  # shape [1, XBLOCK]
    tmp9 = tmp7[None, :]  # shape [1, XBLOCK]

    # Second pass: store normalized values
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base  # shape [R0_BLOCK, 1]
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        # Load: chan_id contiguous across threads -- COALESCED
        tmp10 = tl.load(in_ptr0 + (chan_id + 128*r0_1 + 2032128*batch_id), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 + tmp1
        tmp13 = tmp12 - tmp4
        tmp14 = tl.full([1, 1], 15876.0, tl.float32)
        tmp15 = (tmp8 / tmp14)
        tmp16 = tl.full([1, 1], 1e-05, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tl.full([1, 1], 0.5, tl.float32)
        tmp21 = tmp19 * tmp20
        # Store: chan_id contiguous across threads -- COALESCED
        # out_ptr2 strides: (2032128, 15876, 126, 1)
        # x0 = batch*128 + chan = 128*batch_id + chan_id
        # store address = r0_1 + 15876*(128*batch_id + chan_id)
        tl.store(out_ptr2 + (r0_1 + 15876*(128*batch_id + chan_id)), tmp21, xmask & r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 16384
    r0_numel = 15876
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__native_batch_norm_legit_convolution_div_view_2[grid](arg_0, arg_1, arg_2, 16384, 15876)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 128, 126, 126), (2032128, 1, 16128, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 128, 126, 126), (2032128, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
