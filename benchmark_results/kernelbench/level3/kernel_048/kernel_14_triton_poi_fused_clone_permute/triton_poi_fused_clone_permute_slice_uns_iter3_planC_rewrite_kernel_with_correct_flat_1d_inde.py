import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_permute_slice_unsqueeze_view_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D indexing over all 33,554,432 elements
    # xnumel = 33554432 (total elements), YBLOCK=1 (unused), XBLOCK=block size
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < xnumel
    # Decompose flat output index into 5D coordinates
    # out shape (2048, 2, 8, 16, 64) with strides (16384, 8192, 1024, 64, 1)
    x = offsets % 64                   # output dim4, stride=1
    d = (offsets // 64) % 16           # output dim3, stride=64
    h = (offsets // 1024) % 8          # output dim2, stride=1024
    s = (offsets // 8192) % 2          # output dim1, stride=8192
    n = offsets // 16384               # output dim0, stride=16384
    # Map to input coordinates
    # in shape (16384, 3, 1024) with strides (3072, 1024, 1)
    in_row  = n * 8 + h                # input dim0: 2048*8=16384 total rows
    in_col2 = s                        # input dim1: slice of size-3 dim (s in {0,1})
    in_col3 = d * 64 + x              # input dim2: 16*64=1024 total cols
    in_offset = in_row * 3072 + in_col2 * 1024 + in_col3
    tmp = tl.load(in_ptr0 + in_offset, mask=mask)
    tl.store(out_ptr0 + offsets, tmp, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 524288
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_clone_permute_slice_unsqueeze_view_14[grid](arg_0, arg_1, 524288, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16384, 3, 1024), (3072, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 16, 64, 1), (16384, 8192, 1024, 64, 1, 1), device='cuda:0', dtype=torch.float32)
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
