import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 2048}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_52(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask, other=0.0)
    tl.store(out_ptr0 + (x0 + 512*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 544*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 576*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 608*x1), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + 640*x1), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 62720
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_52[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 62720)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 14, 14), (106624, 1, 7616, 544), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 14, 14), (119168, 1, 8512, 608), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 14, 14), (125440, 1, 8960, 640), device='cuda:0', dtype=torch.float32)
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
