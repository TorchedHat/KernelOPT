import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((x2 % 8)) + 512*x1 + 32768*(x2 // 8)), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 134217728
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8[grid](arg_0, arg_1, 134217728)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2048, 2, 64, 8, 1, 1, 64), (65536, 32768, 512, 64, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
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
