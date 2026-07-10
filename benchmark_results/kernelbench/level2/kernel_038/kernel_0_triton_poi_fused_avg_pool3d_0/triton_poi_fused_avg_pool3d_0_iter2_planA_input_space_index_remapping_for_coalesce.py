import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Input-space decomposition: consecutive threads -> consecutive memory addresses
    ix    = (xindex % 64)              # input W [0,64) -- innermost, stride-1 in memory
    iy    = ((xindex // 64) % 64)      # input H [0,64)
    iz    = ((xindex // 4096) % 32)    # input D [0,32)
    ic_ib = xindex // 131072           # batch*C [0,1024)
    # Single fully-coalesced load (consecutive threads load consecutive floats)
    val = tl.load(in_ptr0 + xindex, xmask)
    # Map to output coordinates (non-overlapping 2x2x2 pool, each input -> exactly one output)
    ox = ix >> 1   # ix // 2
    oy = iy >> 1   # iy // 2
    oz = iz >> 1   # iz // 2
    # Output flat index: shape [32,32,16,32,32], strides [524288,16384,1024,32,1]
    out_idx = ox + 32*oy + 1024*oz + 16384*ic_ib
    # Accumulate 1/8 of each input into its output cell
    tl.atomic_add(out_ptr0 + out_idx, val * 0.125, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 16777216
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_avg_pool3d_0[grid](arg_0, arg_1, 16777216, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 32, 32, 64, 64), (4194304, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
