import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # num_stages=2 (existing baseline)
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        # num_stages=3 — deeper pipeline, hides L2 latency
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        # num_stages=4 — maximum pipeline depth for high-latency DRAM
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
        # YBLOCK=128 configs — y0 = yindex % 128 spans [0..127] exactly per CTA, enabling LDG.128
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        # Large XBLOCK configs — explore wider tiles for potential throughput gains
        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    _yr = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK), 1), YBLOCK)
    yindex = yoffset + _yr[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    mask = ymask & xmask
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=mask, other=0.0, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=mask)

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
