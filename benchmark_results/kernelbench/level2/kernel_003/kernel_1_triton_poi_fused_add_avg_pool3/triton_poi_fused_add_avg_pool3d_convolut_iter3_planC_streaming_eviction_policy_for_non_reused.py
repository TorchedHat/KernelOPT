import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp14 = tmp0 + tmp1 + tmp3 + tmp5 + tmp7 + tmp9 + tmp11 + tmp13
    tmp16 = tmp14 * 0.125
    tmp20 = tmp16 * 0.7071067811865476
    tmp21 = libdevice.erf(tmp20)
    tmp22 = tmp21 + 1.0
    tmp23 = tmp16 * 0.5
    tmp24 = tmp23 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 33554432
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1[grid](arg_0, arg_1, 33554432, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 64, 16, 32, 32), (1048576, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
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
