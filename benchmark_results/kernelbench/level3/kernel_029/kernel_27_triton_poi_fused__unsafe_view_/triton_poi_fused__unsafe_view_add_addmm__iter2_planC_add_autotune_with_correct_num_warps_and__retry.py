import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 376320
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27[grid](arg_0, arg_1, 376320)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 14, 14, 192), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
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
