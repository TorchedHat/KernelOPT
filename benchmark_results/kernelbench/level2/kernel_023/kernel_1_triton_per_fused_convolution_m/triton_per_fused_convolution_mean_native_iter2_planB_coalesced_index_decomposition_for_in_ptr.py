import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 138240
    r0_numel = 440
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 180)
    x4 = xindex // 180
    x1 = ((xindex // 180) % 6)
    x2 = xindex // 1080
    x5 = (xindex % 1080)
    flat = r0_3 + 440 * x0
    d1 = flat // 19800
    d_rem = flat % 19800
    d2 = d_rem // 900
    d_rem2 = d_rem % 900
    d3 = d_rem2 // 30
    d4 = d_rem2 % 30
    ptr0_idx = d4 + 30*d3 + 900*d2 + 19800*d1 + 79200*x4
    tmp0 = tl.load(in_ptr0 + ptr0_idx, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (4*x1 + ((r0_3 + 440*x0) // 19800)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (8*x2 + ((4*x1 + ((r0_3 + 440*x0) // 19800)) // 3)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (8*x2 + ((4*x1 + ((r0_3 + 440*x0) // 19800)) // 3)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (4*x1 + ((r0_3 + 440*x0) // 19800)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (4*x1 + ((r0_3 + 440*x0) // 19800)), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 59400.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x5 + 1088*x2), tmp19, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 138240
    r0_numel = 440
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_mean_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 138240, 440, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 6, 180), (1088, 180, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
