import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 1024,'YBLOCK': 2},  num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4,  num_stages=5),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4,  num_stages=5),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8,  num_stages=5),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8,  num_stages=5),
        triton.Config({'XBLOCK': 1024,'YBLOCK': 2},  num_warps=8,  num_stages=5),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8,  num_stages=6),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8,  num_stages=6),
        triton.Config({'XBLOCK': 1024,'YBLOCK': 1},  num_warps=8,  num_stages=6),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=16, num_stages=5),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=16, num_stages=5),
        triton.Config({'XBLOCK': 1024,'YBLOCK': 1},  num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=8,  num_stages=6),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8,  num_stages=5),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK covers spatial HW (xnumel=31500, stride-1 in OUTPUT) -> COALESCED stores
    # YBLOCK covers NC (ynumel=4096, stride-1 in INPUT) -> small YBLOCK limits register pressure
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.max_contiguous(
                 tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK),
                 XBLOCK)[None, :]   # spatial HW axis, stride-1 in output
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # NC axis, stride-1 in input
    xmask = (xindex < 31500) & (yindex < 4096)
    x0 = xindex          # spatial HW
    y1 = yindex % 128    # channel (stride-1 in input)
    y2 = yindex // 128   # batch
    y3 = yindex          # NC for output (stride-31500 in output)
    tmp0 = tl.load(in_ptr0 + (y1 + 128*x0 + 4032000*y2), xmask, cache_modifier='.ca')
    tl.store(out_ptr0 + (x0 + 31500*y3), tmp0, xmask)

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
