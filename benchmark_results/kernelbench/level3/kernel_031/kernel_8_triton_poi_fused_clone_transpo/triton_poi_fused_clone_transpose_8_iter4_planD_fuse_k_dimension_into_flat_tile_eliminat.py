import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Flat 1D: xindex covers all 4194304 = 8*16384*32 elements
    # x0 = xindex % 32          (k dim, stride-1 in input)
    # x1 = (xindex // 32) % 8   (m/head dim)
    # x2 = xindex // 256         (n/seq dim)
    # Input  (8,16384,32) strides (524288,32,1): offset = x0 + 32*x2 + 524288*x1
    # Output (16384,8,32) strides (256,32,1):   offset = x0 + 32*x1 + 256*x2
    pid = tl.program_id(0)
    xindex = pid * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex % 32           # k: contiguous in input (stride 1)
    x1 = (xindex // 32) % 8   # m: head index
    x2 = xindex // 256         # n: sequence index
    # Vectorization hints: x0 cycles 0..31 contiguously within each warp
    x0_hint = tl.multiple_of(x0, 32)
    x0_hint = tl.max_contiguous(x0_hint, 32)
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x2 + 524288 * x1), mask=xmask, other=0.0)
    tl.store(out_ptr0 + (x0 + 32 * x1 + 256 * x2), tmp0, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 8, 32), (256, 32, 1), device='cuda:0', dtype=torch.float32)
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
