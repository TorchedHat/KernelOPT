import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_29(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x0 + 384*x1), tmp0, xmask, eviction_policy='evict_first')
    tl.store(out_ptr1 + (x0 + 416*x1), tmp0, xmask, eviction_policy='evict_first')
    tl.store(out_ptr2 + (x0 + 448*x1), tmp0, xmask, eviction_policy='evict_first')
    tl.store(out_ptr3 + (x0 + 480*x1), tmp0, xmask, eviction_policy='evict_first')
    tl.store(out_ptr4 + (x0 + 512*x1), tmp0, xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 250880
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_29[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 250880)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 28, 28), (301056, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 28, 28), (351232, 1, 12544, 448), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 28, 28), (376320, 1, 13440, 480), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
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
