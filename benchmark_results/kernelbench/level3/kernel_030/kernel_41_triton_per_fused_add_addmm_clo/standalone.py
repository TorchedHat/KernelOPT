import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_addmm_clone_native_layer_norm_permute_view_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    r0_2 = r0_index
    x0 = (xindex % 784)
    x1 = xindex // 784
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 192*((((x0 % 28)) % 7)) + 1344*(((x0 // 28) % 7)) + 9408*(((x0 % 28)) // 7) + 37632*(x0 // 196) + 150528*x1), r0_mask & xmask, other=0.0)
    tmp19 = tl.load(in_out_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 192, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = tmp2 - tmp12
    tmp21 = tl.full([1, 1], 192.0, tl.float32)
    tmp22 = (tmp18 / tmp21)
    tmp23 = tl.full([1, 1], 1e-05, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp19 + tmp30
    tl.store(in_out_ptr0 + (r0_2 + 192*x3), tmp31, r0_mask & xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 7840
    r0_numel = 192
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_addmm_clone_native_layer_norm_permute_view_41[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 7840, 192, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
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
