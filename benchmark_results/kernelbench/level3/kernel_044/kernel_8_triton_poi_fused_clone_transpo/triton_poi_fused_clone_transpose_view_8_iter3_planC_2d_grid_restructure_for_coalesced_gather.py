import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Grid: program_id(0) = x3 * 8 + x1  (range: 128*8 = 1024)
    #       program_id(1) = x2             (range: 512)
    # Threads: x0 in [0, XBLOCK=128), masked to x0 < 96 (stride-1 in in_ptr0 => COALESCED)
    pid0 = tl.program_id(0)   # encodes (x3, x1)
    pid1 = tl.program_id(1)   # encodes x2
    x3 = pid0 // 8
    x1 = pid0 % 8
    x2 = pid1
    x0 = tl.arange(0, XBLOCK)  # XBLOCK=128 (power of 2), active range [0,96)
    xmask = x0 < 96

    # Load: stride-1 along x0 => coalesced 128-byte cache-line reads
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), mask=xmask, other=0.0)

    # Store: out shape (128,512,8,96) strides (393216,768,96,1)
    # x0 + 96*x1 + 768*x2 + 393216*x3 — also coalesced along x0
    tl.store(out_ptr0 + (x0 + 96*x1 + 768*x2 + 393216*x3), tmp0, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_8[grid](arg_0, arg_1, 50331648, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
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
