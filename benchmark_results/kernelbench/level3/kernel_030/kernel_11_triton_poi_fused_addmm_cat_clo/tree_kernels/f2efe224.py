import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Persistent kernel: each CTA iterates over multiple tiles
    # NUM_TILES = cdiv(3010560, 128) = 23520
    # grid = (132,) = one CTA per SM
    xnumel = 3010560
    NUM_TILES = 23520  # triton.cdiv(3010560, 128)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id < NUM_TILES:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 32)
        x2 = ((xindex // 1568) % 3)
        x1 = ((xindex // 32) % 49)
        x3 = xindex // 4704
        # Load from in_ptr2: the main data tensor
        tmp19 = tl.load(in_ptr2 + (192 + x0 + 32*x2 + 288*x1 + 14112*x3), mask=xmask)
        # Dead-branch elimination:
        # tmp0 = 192 + x0 + 32*x2, x0 in [0,32), x2 in [0,3)
        # => tmp0 in [192, 192+31+64] = [192, 287]
        # => tmp4 = (tmp0 < 96) is ALWAYS FALSE  => in_ptr0 load is dead
        # => tmp9 = (tmp0 >= 96) & (tmp0 < 192) is ALWAYS FALSE => zeros branch is dead
        # => tmp13 = (tmp0 >= 192) is ALWAYS TRUE => in_ptr1 always loads
        tmp16 = tl.load(in_ptr1 + (x0 + 32*x2), mask=xmask, eviction_policy='evict_last')
        tmp20 = tmp16 + tmp19
        tl.store(out_ptr0 + (xoffset + tl.arange(0, XBLOCK)[:]), tmp20, mask=xmask)
        tile_id += num_programs

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
