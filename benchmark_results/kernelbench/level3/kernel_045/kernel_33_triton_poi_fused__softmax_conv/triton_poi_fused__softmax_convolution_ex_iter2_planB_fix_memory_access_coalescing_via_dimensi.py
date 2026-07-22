import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_33(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 8192
    # program_id(1) tiles over ynumel=1024 with step XBLOCK (y0 is fast/thread dim)
    # program_id(0) tiles over xnumel=8192 with step YBLOCK (x2 is slow dim)
    yoffset = tl.program_id(1) * XBLOCK
    yindex = yoffset + tl.arange(0, XBLOCK)[:, None]
    ymask = yindex < ynumel

    xoffset = tl.program_id(0) * YBLOCK
    xindex = xoffset + tl.arange(0, YBLOCK)[None, :]
    xmask = xindex < xnumel

    x2 = xindex                   # spatial, 0..8191  (slow dim)
    y0 = (yindex % 128)           # channel, 0..127   (stride-1 in in_ptr0, FAST dim)
    y1 = yindex // 128            # batch,   0..7

    # Load: y0 is fast dim -> consecutive threads read y0, y0+1, ... -> COALESCED
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 1048576*y1), mask=xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), mask=xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 8192*y0 + 2097152*y1), tmp2, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 1024
    xnumel = 8192
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_33[grid](arg_0, arg_1, arg_2, 1024, 8192)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 128, 32, 256), (1048576, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
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
