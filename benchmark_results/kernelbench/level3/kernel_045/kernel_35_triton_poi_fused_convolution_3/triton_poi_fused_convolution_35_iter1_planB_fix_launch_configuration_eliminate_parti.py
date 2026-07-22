import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flatten the 2D (ynumel=32768, xnumel=9) space into a 1D index
    # With YBLOCK=1, program_id(1) iterates over flat blocks of XBLOCK elements
    # We treat the entire 294912-element space as 1D
    numel = ynumel * xnumel  # 294912
    pid = tl.program_id(0)
    flat_index = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = flat_index < numel
    y3 = flat_index // xnumel
    x2 = flat_index % xnumel
    y0 = y3 % 256
    y1 = y3 // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 32768
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_35[grid](arg_0, arg_1, 32768, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
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
