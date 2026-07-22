import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    CHANNEL: tl.constexpr = 512
    pid = tl.program_id(0)
    # Channel-aligned swizzle: group CTAs by channel to maximize L2 reuse
    # of the 4 evict_last channel tensors (each 512 elements, 8 KB total)
    ctas_per_channel = tl.cdiv(CHANNEL, XBLOCK)   # =1 for XBLOCK>=512
    num_channels = tl.cdiv(xnumel, CHANNEL)        # =490 spatial positions
    channel_id = pid // ctas_per_channel
    intra_channel_id = pid % ctas_per_channel
    swizzled_pid = intra_channel_id * num_channels + channel_id
    xoffset = swizzled_pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)    # single MUFU.RSQ: replaces sqrt_rn+int-div+float-cast
    tmp11 = tmp2 * tmp6            # FMUL: (x-mean) * inv_std
    tmp13 = tmp11 * tmp12          # FFMA: * weight
    tmp15 = tmp13 + tmp14          # FFMA: + bias
    tmp17 = tl.maximum(0.0, tmp15) # float literal: no int->float conversion
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 250880
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 250880)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
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
