import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 512}, num_warps=4),
    triton.Config({'XBLOCK': 512}, num_warps=8),
    triton.Config({'XBLOCK': 1024}, num_warps=4),
    triton.Config({'XBLOCK': 1024}, num_warps=8),
    triton.Config({'XBLOCK': 2048}, num_warps=4),
    triton.Config({'XBLOCK': 2048}, num_warps=8),
], key=['xnumel'])
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 784)
    x2 = xindex // 150528
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 28))) % 7)) + 224*(((4 + (x1 // 28)) % 7)) + 1568*(x0 // 32) + 9408*((4 + ((x1 % 28))) // 7) + 47040*((4 + (x1 // 28)) // 7) + 235200*x2 + ((x0 % 32))), xmask)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 28)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 28))) % 7))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1505280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1505280)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((250, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
