import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,   'YBLOCK': 4}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 4,   'YBLOCK': 4}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 4}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 4}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    # pid(0) = fast dim = Y (channels, stride-1 in output), pid(1) = slow dim = X (spatial)
    yoffset = tl.program_id(0) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK] -- warp-fast
    ymask = yindex < ynumel
    xoffset = tl.program_id(1) * XBLOCK
    xindex = (xoffset + tl.arange(0, XBLOCK))[:, None]  # [XBLOCK, 1] -- warp-slow
    xmask = xindex < xnumel
    x2 = xindex   # [XBLOCK, 1]
    y3 = yindex   # [1, YBLOCK]
    y0 = (yindex % 3)   # 0,1,2 -- stride-1 in output
    y1 = yindex // 3    # image index
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    # consecutive threads span y0=0,1,2 => output addresses differ by 1 float => COALESCED
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
