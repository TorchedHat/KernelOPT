import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'YBLOCK': 4096, 'XBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'YBLOCK': 8192, 'XBLOCK': 64}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_mish_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8388608
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 65536)
    y1 = yindex // 65536
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tl.full([1, 1], 0.5, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1, 1], -1.0, tl.float32)
    tmp13 = tl.maximum(tmp11, tmp12)
    tmp14 = tl.full([1, 1], 1.0, tl.float32)
    tmp15 = tl.minimum(tmp13, tmp14)
    tmp16 = tl.full([1, 1], 2.0, tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (y0 + 65536*x2 + 4194304*y1), tmp17, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 8388608
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_add_convolution_hardtanh_mish_mul_2[grid](arg_0, arg_1, arg_2, 8388608, 64)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 64, 256, 256), (4194304, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
