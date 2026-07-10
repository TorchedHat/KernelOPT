import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel,
                                   YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    C  = 128
    HW = 31500   # H*W = 126*250
    W  = 250
    # XBLOCK covers channel c (stride-1 in input -> coalesced loads)
    xoffset = tl.program_id(0) * XBLOCK
    c = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape [1, XBLOCK]
    xmask = c < C
    # YBLOCK covers flat N*H*W index
    yoffset = tl.program_id(1) * YBLOCK
    flat_hw = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = flat_hw < ynumel
    # Decompose flat_hw -> (n, h, w)
    n   = flat_hw // HW
    rem = flat_hw % HW
    h   = rem // W
    w   = rem % W
    # in_addr: stride (4032000, 1, 32000, 128) -> n*4032000 + c*1 + h*32000 + w*128
    in_addr  = n * 4032000 + c * 1 + h * 32000 + w * 128
    # out_addr: stride (4032000, 31500, 250, 1) -> n*4032000 + c*31500 + h*250 + w*1
    out_addr = n * 4032000 + c * 31500 + h * 250 + w * 1
    mask = xmask & ymask
    tmp0 = tl.load(in_ptr0 + in_addr, mask)
    tl.store(out_ptr0 + out_addr, tmp0, mask)

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
