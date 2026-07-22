import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
], key=['xnumel', 'r0_numel'])
@triton.jit
def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    x0 = (xindex % 512)
    x3 = xindex

    # --- Load ONCE from global memory ---
    tmp0 = tl.load(in_ptr0 + (r0_index + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r0_index + 512*x3), r0_mask, eviction_policy='evict_last', other=0.0)

    # --- Compute scaled+masked scores (keep in registers) ---
    tmp1 = tl.full([1, 1], 0.0, tl.float32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp7 = tl.where(tmp2, tmp6, tmp5)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])

    # --- Single-pass online softmax reduce ---
    _tmp9_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp9_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    _tmp9_max, _tmp9_sum = triton_helpers.online_softmax_combine(_tmp9_max, _tmp9_sum, tmp8, False)
    tmp9, tmp10 = triton_helpers.online_softmax_reduce(_tmp9_max, _tmp9_sum, 1, False)
    tmp9 = tmp9[:, None]
    tmp10 = tmp10[:, None]

    # --- Normalize using register-resident tmp7 (NO second DRAM load) ---
    tmp19 = tmp7 - tmp9
    tmp20 = tl.exp(tmp19)
    tmp21 = tmp20 / tmp10
    tl.store(in_out_ptr0 + (r0_index + 512*x3), tmp21, r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 524288
    r0_numel = 512
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 512), (2097152, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
