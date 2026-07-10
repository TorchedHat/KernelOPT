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
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 128},  num_warps=4),
    ],
    key=['xnumel'],
)
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
    base_idx = tl.max_contiguous(tl.multiple_of(2*x0 + 128*x1 + 8192*x2, 2), 32)
    tmp0  = tl.load(in_ptr0 + base_idx,        mask=xmask, eviction_policy='evict_last')
    tmp1  = tl.load(in_ptr0 + base_idx + 1,    mask=xmask, eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr0 + base_idx + 64,   mask=xmask, eviction_policy='evict_last')
    tmp5  = tl.load(in_ptr0 + base_idx + 65,   mask=xmask, eviction_policy='evict_last')
    tmp7  = tl.load(in_ptr0 + base_idx + 4096, mask=xmask, eviction_policy='evict_last')
    tmp9  = tl.load(in_ptr0 + base_idx + 4097, mask=xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + base_idx + 4160, mask=xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + base_idx + 4161, mask=xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 0.5, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = tl.full([1], 1.0, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 33554432
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1[grid](arg_0, arg_1, 33554432)
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
