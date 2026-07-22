import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: program_id(0) tiles the channel dim (C=96), program_id(1) is the spatial+batch index
    # XBLOCK tiles the channel dimension; each program_id(1) handles one spatial+batch element
    cnumel = 96
    pid_c = tl.program_id(0)   # tile over channel dim
    pid_s = tl.program_id(1)   # one-to-one with spatial+batch index (s)
    c_off = pid_c * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK] — channel offsets
    c_mask = c_off < cnumel   # mask to 96 channels
    s = pid_s  # spatial+batch scalar index (0..7839)
    # Input: flat index = s*96 + c  (input stride over spatial+batch = 96)
    in_idx = s * 96 + c_off
    tmp0 = tl.load(in_ptr0 + in_idx, c_mask)
    # Bias: indexed by channel only
    tmp1 = tl.load(in_ptr1 + c_off, c_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Output: flat index = s*480 + c  (output stride over spatial+batch = 480)
    out_idx = s * 480 + c_off
    tl.store(out_ptr0 + out_idx, tmp2, c_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 752640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_28[grid](arg_0, arg_1, arg_2, 752640)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 96, 28, 28), (75264, 1, 2688, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 96, 28, 28), (376320, 1, 13440, 480), device='cuda:0', dtype=torch.float32)
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
