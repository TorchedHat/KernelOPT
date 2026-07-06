import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    _yarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
    yindex = yoffset + _yarange[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    _xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xoffset + _xarange[None, :]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + ((yindex % 128) + 128*xindex + 4032000*(yindex // 128)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (xindex + 31500*yindex), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
