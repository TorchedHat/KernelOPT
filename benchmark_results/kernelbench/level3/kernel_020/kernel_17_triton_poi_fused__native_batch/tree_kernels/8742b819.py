import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    for tile_id in range(pid, num_tiles, num_sms):
        xoffset = tile_id * XBLOCK
        xindex  = xoffset + tl.arange(0, XBLOCK)[:]
        xmask   = xindex < xnumel
        x0      = xindex % 40
        tmp0  = tl.load(in_out_ptr0 + xindex, xmask)
        tmp1  = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
        tmp3  = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
        tmp2  = tmp0 - tmp1
        tmp5  = tmp3 + tl.full([1], 1e-05, tl.float32)
        tmp10 = tl.rsqrt(tmp5)
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + xindex, tmp15, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 313600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 313600)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
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
