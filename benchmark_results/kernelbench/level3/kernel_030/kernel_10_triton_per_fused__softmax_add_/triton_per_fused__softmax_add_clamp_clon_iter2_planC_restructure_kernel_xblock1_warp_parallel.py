import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 1}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused__softmax_add_clamp_clone_exp_index_log_mul_permute_prepare_softmax_online_sigmoid_sub_unsqueeze_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 49) % 3)
    x0 = (xindex % 49)
    x5 = xindex // 49
    tmp0 = tl.load(in_ptr0 + (r0_3 + 49*x4), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), mask=xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r0_3 + 49*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([1, 1], 4.605170249938965, tl.float32)
    tmp3 = tl.minimum(tmp1, tmp2)
    tmp4 = tl.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tmp7 = tl.full([1, 1], 169, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 169)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp10 < 169")
    tmp12 = tl.load(in_ptr3 + (x1 + 3*tmp10), mask=r0_mask & xmask, eviction_policy='evict_last')
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tl.full([1, 1], 16.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp5 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask & xmask, tmp19, float("-inf"))
    tmp22 = triton_helpers.max2(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp17 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp16 - tmp22
    tmp30 = tl.exp(tmp29)
    tmp31 = (tmp30 / tmp28)
    tl.store(out_ptr2 + (r0_3 + 49*x0 + 2432*x5), tmp31, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 94080
    r0_numel = 49
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_add_clamp_clone_exp_index_log_mul_permute_prepare_softmax_online_sigmoid_sub_unsqueeze_view_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 94080, 49)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((1920, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((169, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((640, 3, 49, 49), (7296, 2432, 49, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
