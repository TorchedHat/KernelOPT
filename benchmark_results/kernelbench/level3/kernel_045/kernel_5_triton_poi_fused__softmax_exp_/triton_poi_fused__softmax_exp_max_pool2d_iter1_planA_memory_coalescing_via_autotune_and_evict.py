import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 256)
    x3 = xindex // 256
    y0 = (yindex % 64)
    y1 = yindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), mask=xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), mask=xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp3 = tl.load(in_ptr0 + (512 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), mask=xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp5 = tl.load(in_ptr0 + (513 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), mask=xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 64*x4 + 524288*y1), tmp6, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 512
    xnumel = 8192
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5[grid](arg_0, arg_1, 512, 8192)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 64, 32, 256), (524288, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
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
