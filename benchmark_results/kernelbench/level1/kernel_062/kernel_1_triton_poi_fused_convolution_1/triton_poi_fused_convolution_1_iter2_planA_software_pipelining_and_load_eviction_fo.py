import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    # num_stages=1 (no pipelining baseline)
    triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=1),
    # num_stages=2 (one prefetch ahead — hides L2/DRAM latency)
    triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 1024,'YBLOCK': 2},  num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=2),
    # num_stages=3 (two prefetches ahead — for high-latency DRAM paths)
    triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 1024,'YBLOCK': 2},  num_warps=8, num_stages=3),
], key=['xnumel', 'ynumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = tl.max_contiguous(tl.multiple_of(xindex, [1, XBLOCK]), [1, XBLOCK])
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    # evict_first: strided loads are never reused — streaming hint frees L2
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask)

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
