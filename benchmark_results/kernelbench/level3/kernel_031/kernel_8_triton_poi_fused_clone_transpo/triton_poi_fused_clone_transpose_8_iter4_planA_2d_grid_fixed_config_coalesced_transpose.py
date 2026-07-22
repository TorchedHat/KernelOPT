import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x1 = tl.program_id(1)  # head axis [0,8) — fixed per block, eliminates stride-2MB jumps
    inner_offset = tl.program_id(0) * XBLOCK
    inner_index = inner_offset + tl.arange(0, XBLOCK)
    inner_mask = inner_index < 524288
    x0 = inner_index % 32   # stride-1 in source — contiguous within warp!
    x2 = inner_index // 32  # stride-32 in source
    # Load: x0 + 32*x2 + 524288*x1 — x1 fixed, x0 stride-1 => fully coalesced
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 524288*x1), inner_mask, eviction_policy='evict_first')
    # Store: original xindex = x0 + 32*x1 + 256*x2 (out shape (16384,8,32), strides (256,32,1))
    tl.store(out_ptr0 + (x0 + 32*x1 + 256*x2), tmp0, inner_mask, eviction_policy='evict_last')

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
