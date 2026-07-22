import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D indexing: XBLOCK is used as the flat block size (YBLOCK ignored)
    numel = ynumel * xnumel  # 98304 * 512 = 50331648
    offset = tl.program_id(0) * XBLOCK
    idx = offset + tl.arange(0, XBLOCK)
    mask = idx < numel
    x2 = idx % xnumel          # column in [0, 512) same as original xindex
    y3 = idx // xnumel         # row in [0, 98304) same as original yindex
    y0 = y3 % 768              # bias slice position unchanged
    y1 = y3 // 768             # batch/head index unchanged
    tmp0 = tl.load(in_ptr0 + (768 + y0), mask=mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), mask=mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 98304
    xnumel = 512
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_addmm_clone_split_transpose_view_2[grid](arg_0, arg_1, arg_2, 98304, 512, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2304,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 96, 512), (393216, 49152, 512, 1), device='cuda:0', dtype=torch.float32)
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
