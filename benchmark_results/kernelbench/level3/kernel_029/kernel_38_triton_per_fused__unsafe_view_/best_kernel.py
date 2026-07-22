import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    r0_index = tl.arange(0, R0_BLOCK)
    r0_mask = r0_index < r0_numel
    r0_2 = r0_index
    for x3 in tl.range(pid, xnumel, num_programs):
        x0 = x3 % 196
        x1 = x3 // 196
        tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), mask=r0_mask, other=0.0)
        tmp1 = tl.load(in_ptr0 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), mask=r0_mask, other=0.0)
        tmp2 = tl.load(in_ptr1 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), mask=r0_mask, other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp14 = tl.where(r0_mask, tmp8, 0.0)
        tmp15 = tl.sum(tmp14, 0).to(tl.float32)
        tmp16 = tl.full([], 384, tl.int32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = (tmp15 / tmp17)
        tmp19 = tmp8 - tmp18
        tmp20 = tmp19 * tmp19
        tmp23 = tl.where(r0_mask, tmp20, 0.0)
        tmp24 = tl.sum(tmp23, 0).to(tl.float32)
        tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, mask=r0_mask)
        tl.store(out_ptr0 + x3, tmp18)
        tl.store(out_ptr1 + x3, tmp24)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 1960
    r0_numel = 384
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1960, 384, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
