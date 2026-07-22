import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # xnumel = cnumel = 60 (number of channels)
    # dim-0 = channel tiles (XBLOCK consecutive channels per block)
    # dim-1 = spatial scalar (program_id(1) in [0, 31360))
    spatial_id = tl.program_id(1)   # spatial index in [0, 31360)
    coffset = tl.program_id(0) * XBLOCK
    cindex = coffset + tl.arange(0, XBLOCK)[:]  # channel indices
    cmask = cindex < xnumel  # xnumel=60, guard out-of-bounds lanes
    # COALESCED: consecutive threads differ by 1 channel = stride-1
    x2 = spatial_id * 60 + cindex
    tmp1  = tl.load(in_ptr0 + cindex, cmask, other=0.0)   # mean[c]
    tmp3  = tl.load(in_ptr1 + cindex, cmask, other=0.0)   # var[c]
    tmp12 = tl.load(in_ptr2 + cindex, cmask, other=1.0)   # weight[c]
    tmp14 = tl.load(in_ptr3 + cindex, cmask, other=0.0)   # bias[c]
    tmp0  = tl.load(in_out_ptr0 + x2, cmask, other=0.0)   # COALESCED load
    tmp2  = tmp0 - tmp1
    tmp4  = tl.full([1], 1e-05, tl.float32)
    tmp5  = tmp3 + tmp4
    tmp6  = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp6
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + x2, tmp17, cmask)  # COALESCED store

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1881600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
