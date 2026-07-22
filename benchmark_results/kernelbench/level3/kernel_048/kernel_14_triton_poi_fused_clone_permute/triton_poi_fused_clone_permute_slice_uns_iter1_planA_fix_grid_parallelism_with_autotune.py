import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 256}, num_warps=8),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4),
    triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=8),
    triton.Config({'XBLOCK': 16,  'YBLOCK': 256}, num_warps=4),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 512}, num_warps=16),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_clone_permute_slice_unsqueeze_view_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = (yindex % 16)
    y1 = ((yindex // 16) % 8)
    y2 = ((yindex // 128) % 2)
    y3 = yindex // 256
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x4 + 1024*y2 + 3072*y1 + 24576*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4 + 64*y5), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 524288
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_clone_permute_slice_unsqueeze_view_14[grid](arg_0, arg_1, 524288, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16384, 3, 1024), (3072, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 16, 64, 1), (16384, 8192, 1024, 64, 1, 1), device='cuda:0', dtype=torch.float32)
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
