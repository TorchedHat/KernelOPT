import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    NUM_SMS: tl.constexpr = 132
    NUM_TILES: tl.constexpr = 29768  # triton.cdiv(3810240, 128)
    pid = tl.program_id(0)
    tile_id = pid
    while tile_id < NUM_TILES:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = ((xindex // 32) % 147)
        x2 = xindex // 4704
        x0 = (xindex % 32)
        x3 = xindex
        tmp0 = (-4) + 7*(((x2 // 9) % 9)) + (((x1 % 49)) // 7)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 56, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = (-4) + 7*((x2 % 9)) + ((((x1 % 49)) % 7))
        tmp6 = tmp5 >= tmp1
        tmp7 = tmp5 < tmp3
        tmp8 = tmp2 & tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tmp9 & tmp7
        tmp11 = tl.load(in_ptr0 + ((-21888) + x0 + 32*(x1 // 49) + 96*((((x1 % 49)) % 7)) + 672*((x2 % 9)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 9) % 9)) + 301056*(x2 // 81)), mask=tmp10 & xmask, other=0.0)
        tmp12 = tl.load(in_ptr1 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tl.load(in_ptr2 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full([1], 96.0, tl.float32)
        tmp16 = (tmp14 / tmp15)
        tmp17 = tl.full([1], 1e-05, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp13 * tmp19
        tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
        tmp26 = tl.where(tmp10, tmp24, tmp25)
        tl.store(out_ptr0 + x3, tmp26, mask=xmask)
        tile_id += NUM_SMS

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 3810240
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3810240, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((810, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
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
