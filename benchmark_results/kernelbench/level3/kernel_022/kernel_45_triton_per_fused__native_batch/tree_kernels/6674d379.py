import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12800
    r0_numel = 49
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        x0 = xindex % 1280
        x1 = xindex // 1280
        tmp1 = tl.load(in_ptr1 + x0, mask=xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + x0, mask=xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr3 + x0, mask=xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr4 + x0, mask=xmask, eviction_policy='evict_last')
        tmp4 = tl.full([XBLOCK], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        inv_std = 1.0 / tmp6
        acc = tl.zeros([XBLOCK], tl.float32)
        for r in tl.static_range(0, 64):
            r_mask = r < 49
            val = tl.load(in_ptr0 + (x0 + 1280 * r + 62720 * x1), mask=xmask & r_mask, other=0.0)
            tmp2 = val - tmp1
            tmp11 = tmp2 * inv_std
            tmp13 = tmp11 * tmp12
            tmp15 = tmp13 + tmp14
            tmp16 = tl.full([XBLOCK], 0, tl.int32)
            tmp17 = tl.maximum(tmp16, tmp15)
            acc += tl.where(r_mask, tmp17, tl.zeros([XBLOCK], tl.float32))
        result = acc / 49.0
        tl.debug_barrier()
        tl.store(in_out_ptr0 + xindex, result, mask=xmask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 12800
    r0_numel = 49
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12800, 49)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 1280, 1, 1), (1280, 1, 12800, 12800), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((490, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
