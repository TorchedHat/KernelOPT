import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_red_fused_convolution_max_pool3d_with_indices_mean_mul_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 135168
    r0_numel = 219
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 66)
    x1 = xindex // 66
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 219*x0
        tmp1 = tl.full([1, 1], 14415, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr0 + (1 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.maximum(tmp3, tmp4)
        tmp6 = tl.load(in_ptr0 + (63 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.maximum(tmp5, tmp6)
        tmp8 = tl.load(in_ptr0 + (64 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.maximum(tmp7, tmp8)
        tmp10 = tl.load(in_ptr0 + (4000 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.maximum(tmp9, tmp10)
        tmp12 = tl.load(in_ptr0 + (4001 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.maximum(tmp11, tmp12)
        tmp14 = tl.load(in_ptr0 + (4063 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.maximum(tmp13, tmp14)
        tmp16 = tl.load(in_ptr0 + (4064 + 2*(((r0_2 + 219*x0) % 31)) + 126*((((r0_2 + 219*x0) // 31) % 31)) + 8000*((((r0_2 + 219*x0) // 961) % 15)) + 124000*x1), xmask & r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.maximum(tmp15, tmp16)
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 135168
    r0_numel = 219
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_convolution_max_pool3d_with_indices_mean_mul_1[grid](arg_0, arg_1, 135168, 219, XBLOCK=128, R0_BLOCK=512)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1984000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 1, 1, 1, 66), (1056, 66, 135168, 135168, 135168, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
