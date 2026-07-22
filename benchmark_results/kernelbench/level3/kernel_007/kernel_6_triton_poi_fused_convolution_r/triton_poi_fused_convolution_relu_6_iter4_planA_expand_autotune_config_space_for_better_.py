import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # Small YBLOCK + XBLOCK=1: maximize CTA count, fill all SMs
        # XBLOCK=1,YBLOCK=16 -> grid=(9,768)=6912 CTAs=52.4 waves
        triton.Config({'XBLOCK': 1,  'YBLOCK': 16},  num_warps=2),
        # XBLOCK=1,YBLOCK=32 -> grid=(9,384)=3456 CTAs=26.2 waves
        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
        # XBLOCK=16,YBLOCK=16 -> grid=(1,768)=768 CTAs=5.8 waves
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=2),
        # XBLOCK=16,YBLOCK=32 -> grid=(1,384)=384 CTAs=2.9 waves (current best)
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
        # Medium YBLOCK configs (unchanged from before)
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
        # Large YBLOCK configs (unchanged from before)
        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
        # NEW: very large YBLOCK with num_warps=16 for high per-CTA occupancy
        # XBLOCK=1,YBLOCK=1024 -> grid=(9,12)=108 CTAs, 16 warps=512 threads/CTA
        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=16),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 1024}, num_warps=16),
        # XBLOCK=1,YBLOCK=2048 -> grid=(9,6)=54 CTAs, 16 warps
        triton.Config({'XBLOCK': 1,  'YBLOCK': 2048}, num_warps=16),
        # XBLOCK=1,YBLOCK=4096 -> grid=(9,3)=27 CTAs, 16 warps
        triton.Config({'XBLOCK': 1,  'YBLOCK': 4096}, num_warps=16),
        # Pipelined: num_stages=2 to overlap memory latency on medium tile
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64 * x2 + 576 * y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
