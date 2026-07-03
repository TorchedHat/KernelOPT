import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # xnumel=96 (channels, stride-1 in in_ptr0), ynumel=774400 (N*HW=256*3025)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # channel c in [0,96)
    xmask = xindex < 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # combined n_hw in [0,774400)
    ymask = yindex < 774400
    c  = xindex          # channel: stride-1 in in_ptr0
    hw = yindex % 3025   # spatial index: stride-96 in in_ptr0
    n  = yindex // 3025  # batch index: stride-290400 in in_ptr0
    # Load: in_ptr0 NHWC strides (290400,1,5280,96) => addr = n*290400 + hw*96 + c*1
    tmp0 = tl.load(in_ptr0 + (c + 96*hw + 290400*n), xmask & ymask, eviction_policy='evict_last')  # COALESCED on c
    tmp1 = tl.load(in_ptr1 + c, xmask, eviction_policy='evict_last')  # bias[c]
    tmp2 = tmp0 + tmp1
    # Store: out_ptr0 NCHW strides (290400,3025,55,1) => addr = n*290400 + c*3025 + hw*1
    tl.store(out_ptr0 + (hw + 3025*c + 290400*n), tmp2, xmask & ymask)

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
