import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 64},  num_warps=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # xnumel = 125440 (spatial: 10 * 112 * 112)
    # pid_c indexes the channel dimension (0..23), no modulo needed
    pid_c = tl.program_id(1)
    pid_x = tl.program_id(0)
    xoffset = pid_x * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    elem_offset = pid_c + 24 * xindex  # channels-last offset
    # Broadcast pid_c to a block-shaped index so we can use xmask on the load
    pid_c_vec = tl.broadcast_to(pid_c + tl.zeros([XBLOCK], dtype=tl.int32), [XBLOCK])
    tmp1  = tl.load(in_ptr0 + pid_c_vec, xmask, eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr1 + pid_c_vec, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + pid_c_vec, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + pid_c_vec, xmask, eviction_policy='evict_last')
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    inv_std = libdevice.rsqrt(tmp5)
    tmp0 = tl.load(in_out_ptr0 + elem_offset, xmask, eviction_policy='evict_first')
    normed = (tmp0 - tmp1) * inv_std
    tmp15 = libdevice.fma(normed, tmp12, tmp14)
    zero = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(zero, tmp15)
    tl.store(in_out_ptr0 + elem_offset, tmp17, xmask, eviction_policy='evict_first')

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
