import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    # xnumel is now spatial_numel = 1960 (= 10 * 14 * 14)
    # pid0 = channel index [0..111], pid1 = spatial tile [0..15]
    x0 = tl.program_id(0)  # channel index, no modulo needed
    spatial_offset = tl.program_id(1) * XBLOCK
    spatial_index = spatial_offset + tl.arange(0, XBLOCK)[:]
    xmask = spatial_index < xnumel  # xnumel = 1960
    # flat index: channel is stride-1 (innermost), spatial stride is 112
    x2 = spatial_index * 112 + x0
    tmp0 = tl.load(in_out_ptr0 + x2, xmask, eviction_policy='evict_first')
    # scalar loads: x0 is a scalar pid, Triton broadcasts to all XBLOCK threads
    tmp1 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + x0, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + x0, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 + 1e-05
    tmp10 = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp10
    tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
    tl.store(in_out_ptr0 + x2, tmp15, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 219520
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
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
