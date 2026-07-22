import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    NUM_SMS = 132
    sm_id = tl.program_id(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = sm_id
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 32)
        x2 = ((xindex // 1568) % 3)
        x1 = ((xindex // 32) % 49)
        x3 = xindex // 4704
        x4 = xindex
        tmp19 = tl.load(in_ptr2 + (192 + x0 + 32*x2 + 288*x1 + 14112*x3), xmask)
        tmp0 = 192 + x0 + 32*x2
        tmp3 = tl.full([1], 96, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (192 + x0 + 32*x2), mask=xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full([1], 192, tl.int64)
        tmp9 = (tmp0 >= tmp3) & (tmp0 < tmp7)
        tmp10 = tl.full([1], 0.0, tl.float32)
        tmp12 = tl.where(tmp9, tmp10, tl.full(tmp10.shape, 0.0, tmp10.dtype))
        tmp13 = tmp0 >= tmp7
        tmp16 = tl.load(in_ptr1 + (x0 + 32*x2), mask=xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.where(tmp9, tmp12, tmp16)
        tmp18 = tl.where(tmp4, tmp5, tmp17)
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr0 + x4, tmp20, xmask)
        tile_id += NUM_SMS

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
