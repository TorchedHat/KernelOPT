import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32,  'XBLOCK': 32}, num_warps=4,  num_stages=1),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 32}, num_warps=4,  num_stages=1),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=8,  num_stages=1),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 32}, num_warps=8,  num_stages=1),
        triton.Config({'YBLOCK': 16,  'XBLOCK': 32}, num_warps=4,  num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 94080
    xnumel = 32
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 147)
    y1 = yindex // 147
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(((y1 // 8) % 8)) + 3136*x2 + 100352*(y0 // 49) + 301056*(y1 // 64) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 96.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + 32*y3), tmp13, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    ynumel = 94080
    xnumel = 32
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 94080, 32)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 3136, 96), (301056, 1, 3136), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((640, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
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
