import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=2, num_stages=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # program_id(0) covers ynumel (channel+batch) — FAST dim for coalescing (stride-1 in input)
    # program_id(1) covers xnumel (spatial) — SLOW dim (stride-128 in input)
    yoffset = tl.program_id(1) * YBLOCK
    yi = tl.arange(0, YBLOCK)
    yindex = yoffset + yi[:, None]   # spatial (stride-128 in input) [YBLOCK, 1]
    ymask = yindex < xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xi = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xoffset + xi[None, :]   # channel+batch (stride-1 in input) [1, XBLOCK]
    xmask = xindex < ynumel
    x0 = (xindex % 128)   # channel within group, stride-1 in input — COALESCED!
    x1 = xindex // 128    # batch index
    y2 = yindex           # spatial index, stride-128 in input
    # Step 1: Coalesced load from global memory (x0 stride-1 across threads)
    tmp0 = tl.load(in_ptr0 + (x0 + 128*y2 + 4032000*x1), xmask & ymask, other=0.0, eviction_policy='evict_last')
    # Step 2: Store to output — output strides (4032000, 31500, 250, 1) -> y2 + 31500*xindex
    tl.store(out_ptr0 + (yindex + 31500*xindex), tmp0, xmask & ymask)

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
