import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 201728
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5[grid](arg_0, arg_1, arg_2, 201728)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((394, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((197, 8, 2, 64), (512, 64, 100864, 1), device='cuda:0', dtype=torch.float32)
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
