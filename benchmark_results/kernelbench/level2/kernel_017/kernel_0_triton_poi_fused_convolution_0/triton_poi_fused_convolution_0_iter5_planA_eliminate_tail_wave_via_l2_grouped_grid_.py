import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 3}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 6}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 6}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 10}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 3}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 3}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 6}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D iteration over NHWC output tensor (numel = ynumel * xnumel = 30 * 50176 = 1505280)
    # Each thread handles one output element at flat index xindex
    # Output (NHWC) flat index: xindex = n*150528 + hw*3 + c
    # Input (NCHW) index: hw + 50176*c + 150528*n
    numel = ynumel * xnumel  # 1505280
    xoffset = tl.program_id(0) * XBLOCK
    x_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xoffset + x_arange
    xmask = xindex < numel
    c   = xindex % 3
    hw  = (xindex // 3) % 50176
    n   = xindex // 150528
    src_idx = hw + 50176 * c + 150528 * n
    tmp0 = tl.load(in_ptr0 + src_idx, xmask)
    tl.store(out_ptr0 + xindex, tmp0, xmask)

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
