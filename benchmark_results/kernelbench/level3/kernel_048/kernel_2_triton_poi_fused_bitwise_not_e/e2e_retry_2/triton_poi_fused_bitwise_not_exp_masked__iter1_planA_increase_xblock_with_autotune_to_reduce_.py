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
        triton.Config({'XBLOCK': 2048}, num_warps=16),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x1 = ((xindex // 3) % 3)
    x3 = xindex // 3
    x2 = xindex // 9
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), xmask, eviction_policy='evict_last')
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp2 & tmp3
    tmp5 = tmp4 == 0
    tmp8 = tmp6 - tmp7
    tmp9 = tl.full([1], float("-inf"), tl.float32)
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x4), tmp11, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 147456
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2[grid](arg_0, arg_1, 147456)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2048, 8, 3), (24, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
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
