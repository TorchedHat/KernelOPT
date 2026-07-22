import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_9(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 512
    y4 = (yindex % 512)
    y0 = (yindex % 64)
    y5 = yindex // 64
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 512*x3 + 32768*y2), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x3 + 4096*y5), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 64*y6), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 2097152
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_clone_permute_unsqueeze_view_9[grid](arg_0, arg_1, arg_2, 2097152, 64)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2048, 128, 8, 64), (65536, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 64, 64, 1, 1), (65536, 32768, 4096, 64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 64, 64, 1), (65536, 32768, 4096, 64, 1, 1), device='cuda:0', dtype=torch.float32)
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
