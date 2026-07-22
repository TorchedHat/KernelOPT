import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    # Swizzle: group CTAs by x2 dimension (160 groups, 74 CTAs each at XBLOCK=128)
    # x2 = xindex // 9408; 9408 / 128 = 73.5 -> 74 CTAs per x2 group
    # Reorder pid so all CTAs with same x2 are consecutive -> better L2 reuse
    pid = tl.program_id(0)
    pid_m = pid % 160   # which x2 group (0..159)
    pid_n = pid // 160  # position within group (0..73)
    xoffset = (pid_m * 74 + pid_n) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 294)
    x2 = xindex // 9408
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 4)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 4)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 4)) + 28*(((x1 % 49)) // 7) + 196*(x2 // 4) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 4)) + 28*(((x1 % 49)) // 7) + 196*(x2 // 4) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 192.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp13 = tl.math.fma(tmp9, tmp10, tmp12)
    tl.store(out_ptr0 + (x3), tmp13, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 1505280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1505280, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((160, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
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
