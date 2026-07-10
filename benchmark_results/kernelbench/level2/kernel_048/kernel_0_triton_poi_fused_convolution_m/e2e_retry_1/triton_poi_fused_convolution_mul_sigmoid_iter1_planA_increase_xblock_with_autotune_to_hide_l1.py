import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110215168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 53816) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.sigmoid(tmp7)
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 110215168
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_mul_sigmoid_tanh_0[grid](arg_0, arg_1, arg_2, arg_3, 110215168)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (861056, 53816, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
