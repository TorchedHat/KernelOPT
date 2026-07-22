import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # program_id(1) = x1 (head dim, 0..7); program_id(0) = flat (x2,x0) index
    x1 = tl.program_id(1)                           # constant per block, size 8
    flat_offset = tl.program_id(0) * XBLOCK
    flat_index = flat_offset + tl.arange(0, XBLOCK)  # covers x2*32 + x0, size 524288
    flat_mask = flat_index < 524288
    x0 = flat_index % 32                             # innermost dim, stride-1 in source
    x2 = flat_index // 32                            # sequence dim (0..16383)
    # Source: shape (8,16384,32), strides (524288,32,1) -> load = x1*524288 + x2*32 + x0
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 524288*x1), flat_mask, eviction_policy='evict_first')
    # Output: shape (16384,8,32), strides (256,32,1) -> store = x2*256 + x1*32 + x0
    tl.store(out_ptr0 + (x0 + 32*x1 + 256*x2), tmp0, flat_mask, eviction_policy='evict_last')

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
