import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4096}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 // 64), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 // 64), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tl.full([1], 64.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 134217728
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_addmm_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 134217728)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((32768, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32768, 64, 1, 1), (64, 1, 2097152, 2097152), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32768, 64, 1, 1), (64, 1, 2097152, 2097152), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
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
