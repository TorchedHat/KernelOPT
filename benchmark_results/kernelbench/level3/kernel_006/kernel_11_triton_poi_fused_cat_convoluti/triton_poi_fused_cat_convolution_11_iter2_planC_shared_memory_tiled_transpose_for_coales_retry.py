import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 50176
    # pid(0) -> channel+batch tiles (ynumel=5120, stride-1 in source)
    # pid(1) -> spatial tiles (xnumel=50176, stride-1 in output)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # shape [XBLOCK, 1] -- channel+batch
    xmask = xindex < ynumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]  # shape [1, YBLOCK] -- spatial
    ymask = yindex < xnumel
    y0 = xindex % 512   # channel within batch (stride-1 in source) -- innermost warp dim for load
    y1 = xindex // 512  # batch index
    y3 = xindex         # flat channel+batch index for output
    x2 = yindex         # spatial index (stride-1 in output) -- innermost warp dim for store
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 25690112*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 50176*y3), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 5120
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_cat_convolution_11[grid](arg_0, arg_1, 5120, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 512, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 512, 224, 224), (25690112, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
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
