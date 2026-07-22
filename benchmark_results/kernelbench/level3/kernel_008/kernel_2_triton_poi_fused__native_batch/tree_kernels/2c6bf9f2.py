import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    # xnumel here is snumel = 125440 (spatial elements per channel)
    c0 = tl.program_id(0)
    soffset = tl.program_id(1) * XBLOCK
    sindex = soffset + tl.arange(0, XBLOCK)[:]
    smask = sindex < xnumel
    flat_idx = c0 + sindex * 24
    tmp1 = tl.load(in_ptr0 + c0)
    tmp3 = tl.load(in_ptr1 + c0)
    tmp12 = tl.load(in_ptr2 + c0)
    tmp14 = tl.load(in_ptr3 + c0)
    tmp0 = tl.load(in_out_ptr0 + flat_idx, smask)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp6
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.float32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + flat_idx, tmp17, smask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 24, 112, 112), (301056, 1, 2688, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
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
