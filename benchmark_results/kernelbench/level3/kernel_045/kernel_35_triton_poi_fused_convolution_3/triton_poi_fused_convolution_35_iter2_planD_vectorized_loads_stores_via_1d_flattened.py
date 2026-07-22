import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flatten 2D iteration space into 1D for stride-1 sequential loads
    # numel = ynumel * xnumel = 32768 * 9 = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    numel = 294912
    xmask = xindex < numel
    # Recover 2D indices from flat index
    x2 = xindex % 9          # spatial kernel index (0..8)
    y3 = xindex // 9         # batch*channel index (0..32767)
    y0 = y3 % 256            # output channel
    y1 = y3 // 256           # batch
    # Stride-1 sequential load from contiguous source tensor
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    # Output index formula unchanged — encodes target permuted layout
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 32768
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_35[grid](arg_0, arg_1, 32768, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
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
