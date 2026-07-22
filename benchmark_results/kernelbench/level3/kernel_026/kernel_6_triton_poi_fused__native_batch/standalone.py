import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 3136
    ynumel = 600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    x3 = xindex
    y2 = yindex // 60
    y4 = (yindex % 60)
    y0 = (yindex % 20)
    y1 = ((yindex // 20) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 60*x3 + 188160*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 + 1e-05
    tmp6 = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp6
    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 188160*y2), tmp15, xmask & ymask)


def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    ynumel = 600
    xnumel = 3136
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 600, 3136, YBLOCK=8, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 20, 3, 56, 56), (188160, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
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
