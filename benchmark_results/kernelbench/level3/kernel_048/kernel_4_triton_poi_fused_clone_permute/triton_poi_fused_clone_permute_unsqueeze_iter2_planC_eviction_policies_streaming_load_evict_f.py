import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 128},  num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 64)
    x2 = ((xindex // 1024) % 8)
    x3 = xindex // 8192
    x4 = xindex
    load_idx = x0 + 16*x2 + 128*x1 + 8192*x3
    load_idx = tl.multiple_of(load_idx, 16)
    load_idx = tl.max_contiguous(load_idx, 16)
    tmp0 = tl.load(in_ptr0 + load_idx, xmask, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x4), tmp0, xmask, eviction_policy='evict_last')
    tl.store(out_ptr1 + (x4), tmp0, xmask, eviction_policy='evict_last')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 33554432
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_permute_unsqueeze_view_4[grid](arg_0, arg_1, arg_2, 33554432)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 64, 16, 1, 1), (16384, 8192, 1024, 16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
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
