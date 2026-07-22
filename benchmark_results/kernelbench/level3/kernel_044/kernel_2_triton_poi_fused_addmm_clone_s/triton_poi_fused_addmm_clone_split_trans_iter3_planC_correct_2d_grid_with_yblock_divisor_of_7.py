import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 8},  num_warps=4),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 32}, num_warps=4),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 4}, num_warps=4),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=8),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=8),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 8}, num_warps=8),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 4}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 768)
    x2 = xindex
    y1 = yindex // 768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0), mask=ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), mask=xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 98304
    xnumel = 512
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_addmm_clone_split_transpose_view_2[grid](arg_0, arg_1, arg_2, 98304, 512)
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
