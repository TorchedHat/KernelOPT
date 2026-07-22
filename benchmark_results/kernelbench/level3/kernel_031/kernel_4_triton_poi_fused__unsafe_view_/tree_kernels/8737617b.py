import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yarange = tl.max_contiguous(tl.multiple_of(yoffset + tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
    yindex = yarange[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xarange[:, None]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (4194304 + y0 + 256*x1), ymask, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp0, ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 256
    xnumel = 16384
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4[grid](arg_0, arg_1, 256, 16384, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((3, 16384, 2, 128), (4194304, 256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 32, 16384), (524288, 16384, 1), device='cuda:0', dtype=torch.float32)
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
