import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    # Flat 1D indexing: treat YBLOCK*XBLOCK as block size, use only program_id(0)
    BLOCK_SIZE: tl.constexpr = YBLOCK * XBLOCK
    numel: tl.constexpr = ynumel * xnumel
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    # Recover x3 and y5 from flat index (flat = y5*16 + x3)
    x3 = offsets % 16
    y5 = offsets // 16
    # Derived sub-indices -- identical to original
    y0 = y5 % 64
    y1 = (y5 // 64) % 8
    y2 = y5 // 512
    y4 = y5 // 64
    # Loads -- identical address expressions to original
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y1 + 128*y0 + 8192*y2), mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (63 + 64*y4), mask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + y5, mask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tmp0 * tmp4
    # Stores -- identical address expressions to original
    tl.store(out_ptr0 + (y0 + 64*x3 + 1024*y4), tmp0, mask)
    tl.store(out_ptr1 + (x3 + 16*y5), tmp5, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 2097152
    xnumel = 16
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5[grid](arg_0, arg_1, arg_2, arg_3, 2097152, 16, YBLOCK=8, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 16, 1, 64, 1), (16384, 8192, 1024, 64, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
