import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Each CTA encodes one (x2, x1) outer pair; XBLOCK=32 threads cover x0 contiguously
    pid = tl.program_id(0)   # range [0, 8*16384) = [0, 131072)
    x1 = pid % 16384
    x2 = pid // 16384
    x0 = tl.arange(0, XBLOCK)  # XBLOCK=32, contiguous thread lanes [0..31]

    # Output index: strides [524288, 32, 1] - x0 is contiguous (stride=1)
    x3 = x2 * 524288 + x1 * 32 + x0
    xmask = x3 < xnumel

    # col = x0 + 32*x2: for fixed x2, varies contiguously with x0
    # For x2 in [0..7]: col in [0..255], col//128 in {0,1} - CONSTANT per warp
    # col%128 = x0 + 32*x2 - 128*(col//128) increments by 1 with x0 => stride-1 load
    col = x0 + 32 * x2
    in_idx = 384 * (col // 128) + 768 * x1 + 768 * (col // 256) + (col % 128)

    tmp0 = tl.load(in_ptr0 + in_idx, xmask)           # COALESCED: stride-1 in last dim
    tmp1 = tl.load(in_ptr1 + (col % 128), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.1767766952966369, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)              # COALESCED: stride-1 store

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2[grid](arg_0, arg_1, arg_2, 4194304, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
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
