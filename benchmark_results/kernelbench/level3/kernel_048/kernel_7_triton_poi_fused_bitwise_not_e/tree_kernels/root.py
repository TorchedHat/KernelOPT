import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y5 = yindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 8)
    y2 = ((yindex // 512) % 2)
    y3 = yindex // 1024
    y7 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x4 + 64*y5), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (y0 + 64*y2 + 128*y1 + 1024*y3), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x4 + 64*y2 + 128*y1 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = x4 + ((-1)*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1, 1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5 == 0
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp11 = tl.where(tmp6, tmp10, tmp9)
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp0 * tmp12
    tl.store(out_ptr0 + (x4 + 64*y1 + 512*y0 + 32768*y7), tmp13, xmask & ymask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 2097152
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7[grid](arg_0, arg_1, arg_2, 2097152, 64, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 64, 8, 1, 1, 64), (65536, 32768, 512, 64, 64, 64, 1), device='cuda:0', dtype=torch.float32)
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
