import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Grid: axis-0 = N-tiles cdiv(524288, XBLOCK), axis-1 = 8 heads
    pid_n = tl.program_id(0)   # tile over N*K=524288 combined elements
    pid_m = tl.program_id(1)   # head index 0..7, FIXED scalar per block
    n_index = pid_n * XBLOCK + tl.arange(0, XBLOCK)  # 0..524287
    nmask = n_index < 524288
    x0 = n_index % 32    # k dim: 0..31
    x2 = n_index // 32   # seq dim: 0..16383
    # load_offset = x0 + 32*x2 + 524288*pid_m = n_index + 524288*pid_m (algebraically simplified)
    # All threads share pid_m => stride-1 contiguous loads => FULLY COALESCED
    tmp0 = tl.load(in_ptr0 + n_index + 524288 * pid_m, mask=nmask, other=0.0)
    # store_offset = x0 + 32*pid_m + 256*x2
    tl.store(out_ptr0 + (x0 + 32 * pid_m + 256 * x2), tmp0, mask=nmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304, XBLOCK=128)
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
