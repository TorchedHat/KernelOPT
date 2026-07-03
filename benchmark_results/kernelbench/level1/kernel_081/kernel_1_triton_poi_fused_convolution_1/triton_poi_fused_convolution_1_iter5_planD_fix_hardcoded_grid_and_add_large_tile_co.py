import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    # Original configs with num_stages=2
    triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=2),
    # num_stages=3 variants
    triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
    # num_stages=4 variants
    triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=4),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=4),
    # num_stages=5 variants (deepest pipeline for max latency hiding)
    triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=5),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=5),
    # New: num_warps=2 configs with small tiles
    triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=2, num_stages=4),
    # New: num_warps=16 configs with large tiles (num_stages=1 for SMEM budget)
    triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=16, num_stages=1),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=16, num_stages=1),
    # New large-tile configs appended per optimization plan
    triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=16, num_stages=2),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=16, num_stages=2),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=16, num_stages=3),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=16, num_stages=3),
    triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=8,  num_stages=2),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=8,  num_stages=2),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4,  num_stages=3),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4,  num_stages=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK

    # Phase 1: Coalesced LOAD - Y-threads walk stride-1 channel dim
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # [1, XBLOCK]
    ymask = yindex < ynumel
    xmask = (xindex < xnumel) & ymask
    y0 = yindex % 128        # channel, stride-1 in input
    y1 = yindex // 128       # batch
    x2 = xindex              # spatial
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1),
                   mask=xmask, other=0.0)  # [YBLOCK, XBLOCK]

    # Phase 2: Transpose in shared memory
    tmp0_T = tl.trans(tmp0)  # [XBLOCK, YBLOCK]

    # Phase 3: Coalesced STORE - X-threads walk stride-1 spatial dim in output
    xindex2 = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    yindex2 = yoffset + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK]
    xmask2 = (xindex2 < xnumel) & (yindex2 < ynumel)
    tl.store(out_ptr0 + (xindex2 + 31500*yindex2), tmp0_T, mask=xmask2)

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
