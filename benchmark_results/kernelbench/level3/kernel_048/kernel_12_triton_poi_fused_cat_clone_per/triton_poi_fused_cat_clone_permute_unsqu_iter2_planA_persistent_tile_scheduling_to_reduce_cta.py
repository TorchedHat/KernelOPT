import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Persistent kernel: each CTA loops over multiple tiles
    # NUM_PROGRAMS = 528 (132 SMs * 4 tiles/SM), XBLOCK = 1024
    # stride = 528 * 1024 = 540672; tiles per CTA = 50331648 / 540672 = 93 exactly
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    num_programs = tl.num_programs(0)
    stride = num_programs * XBLOCK  # 528 * 1024 = 540672
    while xoffset < xnumel:  # loops 93 times per CTA
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = tl.full([XBLOCK], True, tl.int1)  # safe: 50331648 % 540672 == 0
        x1 = ((xindex // 1024) % 3)
        x0 = (xindex % 1024)
        x2 = ((xindex // 3072) % 8)
        x3 = xindex // 24576
        x4 = xindex
        tmp0 = x1
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.full([1], 0.0, tl.float32)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1], 3, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), mask=tmp8, other=0.0)
        tmp12 = tl.where(tmp4, tmp7, tmp11)
        tl.store(out_ptr0 + (x4), tmp12, mask=xmask)
        xoffset += stride

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12[grid](arg_0, arg_1, 50331648, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32768, 64, 16), (1024, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
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
