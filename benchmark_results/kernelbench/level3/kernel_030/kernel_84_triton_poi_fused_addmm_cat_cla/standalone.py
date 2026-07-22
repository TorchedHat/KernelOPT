import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_84(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 490
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x5 = xindex
    x2 = (xindex % 32)
    x3 = xindex // 32
    y4 = yindex
    y0 = (yindex % 49)
    y1 = yindex // 49
    tmp19 = tl.load(in_ptr2 + (768 + x5 + 2304*y4), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x3 + 24*y4), xmask & ymask, eviction_policy='evict_last')
    tmp0 = 768 + x5
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(768 + x2 + 32*x3, [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 1536, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.full([1, 1], 0.0, tl.float32)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp0 >= tmp7
    tmp14 = tl.full([1, 1], 2304, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (tl.broadcast_to((-768) + x2 + 32*x3, [YBLOCK, XBLOCK])), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp12, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tmp20 = tmp18 + tmp19
    tmp22 = tl.sqrt_rn(tmp21)
    tmp23 = tl.full([1, 1], 1e-12, tl.float32)
    tmp24 = tl.maximum(tmp22, tmp23)
    tmp25 = (tmp20 / tmp24)
    tl.store(out_ptr0 + (y0 + 49*x5 + 37632*y1), tmp25, xmask & ymask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    ynumel = 490
    xnumel = 768
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_84[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 490, 768, YBLOCK=8, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 24, 49, 1), (1176, 1, 24, 11760), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 24, 32, 49), (37632, 1568, 49, 1), device='cuda:0', dtype=torch.float32)
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
