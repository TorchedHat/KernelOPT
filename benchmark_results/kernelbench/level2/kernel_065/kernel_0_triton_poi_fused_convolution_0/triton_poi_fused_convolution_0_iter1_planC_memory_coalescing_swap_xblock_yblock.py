import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 256}, num_warps=8, num_stages=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    # XBLOCK now covers ynumel (channel dim=30) — contiguous in output
    # YBLOCK now covers xnumel (spatial dim=50176)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK], covers ynumel
    xmask = xindex < ynumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1], covers xnumel
    ymask = yindex < xnumel
    y2 = yindex   # spatial index (old x2)
    x3 = xindex   # combined channel index (old y3)
    x0 = xindex % 3    # old y0
    x1 = xindex // 3   # old y1
    tmp0 = tl.load(in_ptr0 + (y2 + 50176*x3), mask=ymask & xmask, eviction_policy='evict_last')
    # x0 varies fastest across warp threads -> contiguous output writes
    tl.store(out_ptr0 + (x0 + 3*y2 + 150528*x1), tmp0, mask=ymask & xmask)

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
