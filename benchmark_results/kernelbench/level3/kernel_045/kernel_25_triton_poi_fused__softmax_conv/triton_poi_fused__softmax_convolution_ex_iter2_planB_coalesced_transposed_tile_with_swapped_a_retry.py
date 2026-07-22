import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_25(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # program_id(0) tiles y0 (channel dim, stride-1 in in_ptr0 -> coalesced loads)
    # program_id(1) tiles the combined (y1*512 + x2) dimension
    y0_offset = tl.program_id(0) * XBLOCK
    y0 = y0_offset + tl.arange(0, XBLOCK)[None, :]   # shape [1, XBLOCK]
    y0_mask = y0 < 512

    pid1 = tl.program_id(1)
    x2_y1_offset = pid1 * YBLOCK
    x2_y1 = x2_y1_offset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    x2 = x2_y1 % 512
    y1 = x2_y1 // 512
    x2_y1_mask = x2_y1 < 4096

    combined_mask = y0_mask & x2_y1_mask

    # Load from in_ptr0: consecutive threads differ in y0 (stride=1) -> coalesced
    tmp0 = tl.load(in_ptr0 + (y0 + 512 * x2 + 262144 * y1), combined_mask, eviction_policy='evict_last')
    # Load bias: indexed by y0, broadcast across x2/y1 rows -> L1 reuse
    tmp1 = tl.load(in_ptr1 + y0, y0_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512 * y0 + 524288 * y1), tmp2, combined_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 4096
    xnumel = 512
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_25[grid](arg_0, arg_1, arg_2, 4096, 512, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 512, 8, 64), (262144, 1, 32768, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 512, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
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
