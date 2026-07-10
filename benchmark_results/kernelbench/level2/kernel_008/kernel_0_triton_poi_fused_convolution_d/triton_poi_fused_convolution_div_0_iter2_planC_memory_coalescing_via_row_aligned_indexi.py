import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_div_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Iterate over OUTPUT flat index space
    # Output strides: (867328, 54208, 3872, 62, 1) for shape (128,16,14,62,62)
    # Input  strides: (861056, 53816, 3844, 62, 1) for shape (128,16,14,62,62)
    xnumel = 110215168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    # Decompose OUTPUT flat index into multi-dim coordinates
    tmp_d0 = xindex // 867328                          # batch (0..127)
    tmp_rem0 = xindex % 867328
    tmp_d1 = tmp_rem0 // 54208                         # channel (0..15) -> bias index
    tmp_rem1 = tmp_rem0 % 54208
    tmp_d2 = tmp_rem1 // 3872                          # depth (0..13)
    tmp_rem2 = tmp_rem1 % 3872
    tmp_d3 = tmp_rem2 // 62                            # height (0..61)
    tmp_d4 = tmp_rem2 % 62                             # width (0..61)
    # Reconstruct INPUT flat index using input strides
    x4_in = (tmp_d0 * 861056 + tmp_d1 * 53816 + tmp_d2 * 3844 + tmp_d3 * 62 + tmp_d4)
    x2 = tmp_d1                                        # channel index for bias
    tmp0 = tl.load(in_ptr0 + x4_in, None)
    tmp1 = tl.load(in_ptr1 + x2, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + xindex, tmp4, None)            # COALESCED: consecutive xindex -> consecutive output addr

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 110215168
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_div_0[grid](arg_0, arg_1, arg_2, 110215168, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (861056, 53816, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 14, 62, 62), (867328, 54208, 3872, 62, 1), device='cuda:0', dtype=torch.float32)
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
