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
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=5),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=5),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK = fast dim = N*C=4096 (channel stride-1 in input -> coalesced loads)
    # YBLOCK = slow dim = H*W=31500 (spatial)
    ynumel = 4096  # N*C (used as xnumel in new layout)
    xnumel = 31500  # H*W (used as ynumel in new layout)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # N*C fast dim
    xmask = xindex < ynumel
    c0 = xindex % 128    # channel (stride-1 in input -> COALESCED)
    n1 = xindex // 128   # batch
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # H*W slow dim
    ymask = yindex < xnumel
    s2 = yindex          # spatial flat index (h*250+w)
    # Load: c0 stride-1, n1 stride-4032000, s2 stride-128 in input -> coalesced on c0
    tmp0 = tl.load(in_ptr0 + (c0 + 4032000*n1 + 128*s2), xmask & ymask)
    # Store: out stride (4032000,31500,250,1) -> n1*4032000 + c0*31500 + s2
    tl.store(out_ptr0 + (s2 + 31500*xindex), tmp0, xmask & ymask)

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
