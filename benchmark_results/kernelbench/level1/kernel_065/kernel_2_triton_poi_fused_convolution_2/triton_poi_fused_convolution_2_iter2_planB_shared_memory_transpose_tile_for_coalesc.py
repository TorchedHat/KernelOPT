import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK
    y_local = tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1]
    x_local = tl.arange(0, XBLOCK)[None, :]   # [1, XBLOCK]
    yindex = yoffset + y_local   # [YBLOCK, 1]
    xindex = xoffset + x_local   # [1, XBLOCK]
    ymask = yindex < ynumel      # [YBLOCK, 1]
    xmask = (xindex < xnumel) & ymask  # [YBLOCK, XBLOCK] combined mask
    y0 = (yindex % 96)           # channel index, stride-1 in in_ptr0
    y1 = yindex // 96            # batch index
    y3 = yindex                  # flat y for output
    # Phase 1: COALESCED LOAD - y0 (stride-1) varies along YBLOCK axis
    # For a fixed x_local column, consecutive y_local threads access
    # consecutive memory addresses (y0 stride=1) -> coalesced
    val = tl.load(in_ptr0 + (y0 + 96*xindex + 290400*y1), mask=xmask, eviction_policy='evict_last')
    # Bias add (broadcast over x dimension)
    bias = tl.load(in_ptr1 + y0, mask=ymask, eviction_policy='evict_last')
    val = val + bias
    # Phase 2: COALESCED STORE - xindex (stride-1 in out_ptr0) varies along XBLOCK axis
    tl.store(out_ptr0 + (xindex + 3025*y3), val, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
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
