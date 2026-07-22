import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 256},  num_warps=8),
        triton.Config({'XBLOCK': 128},  num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 1176)
    x2 = xindex // 37632
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 376320
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 376320)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1176, 32), (37632, 32, 1), device='cuda:0', dtype=torch.float32)
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
