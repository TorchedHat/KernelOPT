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
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # X axis = batch*channel (4096), stride-1 in in_ptr0 -> FAST axis (coalesced loads)
    # Y axis = spatial H*W  (31500), stride-128 in in_ptr0 -> SLOW axis
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK], batch*channel index
    xmask = xindex < 4096  # 4096 = 32*128, exact, always True but keep for safety
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1], spatial index
    ymask = yindex < 31500
    x0 = xindex % 128   # channel (0..127), stride=1 in in_ptr0
    x1 = xindex // 128  # batch   (0..31)
    y2 = yindex          # spatial (0..31499)
    # Load: consecutive xindex threads differ by 1 in x0 (stride=1 in in_ptr0) -> COALESCED
    tmp0 = tl.load(in_ptr0 + (x0 + 128*y2 + 4032000*x1), xmask & ymask)
    # Store: out_ptr0 has stride (4032000, 31500, 250, 1)
    # out address = batch*4032000 + channel*31500 + spatial
    #             = x1*4032000 + x0*31500 + y2
    #             = (x1*128 + x0)*31500 + y2  [since x1*4032000 + x0*31500 = (x1*128+x0)*31500]
    # Wait: x1*4032000 + x0*31500 = x1*128*31500 + x0*31500 = (x1*128+x0)*31500 = xindex*31500
    # So: out address = y2 + 31500*xindex  ✓
    tl.store(out_ptr0 + (y2 + 31500*xindex), tmp0, xmask & ymask)

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
