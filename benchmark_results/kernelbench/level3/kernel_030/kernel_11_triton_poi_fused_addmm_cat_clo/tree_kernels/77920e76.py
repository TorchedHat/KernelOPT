import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # PROOF: x0 in [0,31], x2 in [0,2] => tmp0 = 192+x0+32*x2 in [192,287]
    # => tmp0 < 96 is ALWAYS False, (tmp0>=96)&(tmp0<192) is ALWAYS False,
    # => tmp0 >= 192 is ALWAYS True => only in_ptr1 branch is ever taken
    # => in_ptr0 is never loaded; entire tl.where chain collapses to in_ptr1 load
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x2 = ((xindex // 1568) % 3)
    x1 = ((xindex // 32) % 49)
    x3 = xindex // 4704
    # Unconditional bias load from in_ptr1 (always the active branch)
    tmp_bias = tl.load(in_ptr1 + (x0 + 32*x2), xmask, eviction_policy='evict_last')
    # Main tensor load (unchanged offset)
    tmp_main = tl.load(in_ptr2 + (192 + x0 + 32*x2 + 288*x1 + 14112*x3), xmask)
    tl.store(out_ptr0 + xindex, tmp_bias + tmp_main, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11[grid](arg_0, arg_1, arg_2, arg_3, 3010560, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((640, 3, 49, 32), (4704, 1568, 32, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
