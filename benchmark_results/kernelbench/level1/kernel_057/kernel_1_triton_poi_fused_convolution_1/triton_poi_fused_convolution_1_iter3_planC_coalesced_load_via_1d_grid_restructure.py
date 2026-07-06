import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 31500 (HW = H*W = 126*250), xnumel = 4096 (N*C = 32*128)
    # XBLOCK = 128 = C, so xindex covers channels (stride=1 in input) -> COALESCED loads
    # YBLOCK covers HW positions (stride=128 in input) -> each row is one hw position
    ynumel = 31500
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel

    # xindex covers N*C: x0 = channel (0..127), x1 = batch (0..31)
    x0 = xindex % 128   # channel index, stride=1 in input
    x1 = xindex // 128  # batch index
    x3 = xindex         # flat N*C index

    # yindex covers HW positions
    y2 = yindex         # flat HW index

    # INPUT channels-last strides: in_ptr0[n,c,h,w] = n*4032000 + c + hw*128
    # COALESCED: consecutive threads differ in x0 (stride=1 in input)
    tmp0 = tl.load(in_ptr0 + (x0 + 128*y2 + 4032000*x1), xmask & ymask)

    # OUTPUT contiguous strides: out_ptr0[n,c,h,w] = n*4032000 + c*31500 + hw
    # Stores: consecutive threads differ in x0 -> stride=31500 (non-coalesced)
    # but hw (y2) is the fast-varying in output -> coalesced across YBLOCK rows
    tl.store(out_ptr0 + (y2 + 31500*x3), tmp0, xmask & ymask)

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
