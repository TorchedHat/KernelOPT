import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    total_tiles = xnumel // XBLOCK  # exact: 12582912 divisible by 128,256,512
    for tile_id in tl.range(pid, total_tiles, grid_size):
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = (xindex % 128)
        x1 = ((xindex // 128) % 32768)
        x2 = xindex // 4194304
        tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1))
        tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + xindex, tmp2)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 12582912
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3[grid](arg_0, arg_1, arg_2, 12582912)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((3, 16384, 2, 128), (4194304, 256, 128, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
