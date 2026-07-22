import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8, 'YBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 512}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 128)
    y4 = yindex // 128
    y2 = yindex // 16384
    y5 = (yindex % 16384)
    tmp0 = tl.load(in_ptr0 + (x3 + 128*y0 + 32768*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x3 + 128*y0 + 32768*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16384 + x3 + 128*y0 + 32768*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16448 + x3 + 128*y0 + 32768*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 4.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tl.store(out_ptr0 + (y5 + 16384*x3 + 1048576*y2), tmp8, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 2097152
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2[grid](arg_0, arg_1, 2097152, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8388608, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
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
