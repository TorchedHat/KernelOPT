import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 64,   'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 128,  'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 16}, num_warps=8),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 16}, num_warps=8),
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 8)
    y2 = yindex // 512
    y4 = yindex // 64
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y1 + 128*y0 + 8192*y2), mask=xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (63 + 64*y4), mask=ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y5), mask=ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (y0 + 64*x3 + 1024*y4), tmp0, mask=xmask & ymask)
    tl.store(out_ptr1 + (x3 + 16*y5), tmp5, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 2097152
    xnumel = 16
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5[grid](arg_0, arg_1, arg_2, arg_3, 2097152, 16)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 16, 1, 64, 1), (16384, 8192, 1024, 64, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
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
