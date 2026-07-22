import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 192*x3), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*((((x0 % 28)) % 7)) + 224*(((x0 // 28) % 7)) + 1568*(r0_2 // 32) + 9408*(((x0 % 28)) // 7) + 37632*(x0 // 196) + 150528*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((x0 // 28) % 7)) + 49*(r0_2 // 32) + ((((x0 % 28)) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 192*x3), mask=r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 192*x3), tmp8, mask=r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, mask=xmask)
    tl.store(out_ptr1 + (x3), tmp24, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 7840
    r0_numel = 192
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 7840, 192)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((160, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
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
