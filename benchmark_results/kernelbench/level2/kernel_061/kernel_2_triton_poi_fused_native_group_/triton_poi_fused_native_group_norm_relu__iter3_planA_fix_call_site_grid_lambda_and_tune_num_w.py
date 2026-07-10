import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80494592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x6 = xindex // 39304
    x2 = ((xindex // 39304) % 128)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp3 = tl.load(in_ptr1 + (x6 // 16), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x6 // 16), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0)
    tmp4 = tmp2 - tmp3
    tmp7 = tmp5 * (1.0 / 628864.0)
    tmp8 = tl.full([1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x4), tmp15, xmask)

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
