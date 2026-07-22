import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=16),
        triton.Config({'XBLOCK': 4096}, num_warps=8),
        triton.Config({'XBLOCK': 4096}, num_warps=16),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_transpose_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_8[grid](arg_0, arg_1, 50331648)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
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
