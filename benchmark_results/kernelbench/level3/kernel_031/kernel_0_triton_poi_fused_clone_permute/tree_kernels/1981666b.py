import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_clone_permute_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    y_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK) + yoffset, YBLOCK), YBLOCK)
    yindex = y_arange[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    x_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK) + xoffset, XBLOCK), XBLOCK)
    xindex = x_arange[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16384*x1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 16384
    xnumel = 256
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_clone_permute_view_0[grid](arg_0, arg_1, 16384, 256)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
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
