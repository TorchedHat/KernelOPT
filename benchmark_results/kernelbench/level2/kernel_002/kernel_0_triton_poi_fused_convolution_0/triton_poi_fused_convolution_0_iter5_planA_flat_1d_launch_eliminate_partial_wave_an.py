import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=2, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel is passed as numel = 30 * 50176 = 1505280 (total elements)
    # xnumel is passed as 50176 (spatial elements per channel)
    # YBLOCK is always 1 in all configs - used only to satisfy signature
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    mask = xindex < ynumel
    # Decompose flat index into spatial (x2) and channel+batch (y3)
    x2 = xindex % xnumel          # spatial index in [0, 50176)
    y3 = xindex // xnumel         # combined channel+batch index in [0, 30)
    y0 = y3 % 3                   # channel within group [0, 3)
    y1 = y3 // 3                  # batch index [0, 10)
    # Load: input NCHW contiguous stride=(150528,50176,224,1)
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), mask, eviction_policy='evict_first')
    # Store: output NHWC-like stride=(150528,1,672,3)
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, mask)

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
