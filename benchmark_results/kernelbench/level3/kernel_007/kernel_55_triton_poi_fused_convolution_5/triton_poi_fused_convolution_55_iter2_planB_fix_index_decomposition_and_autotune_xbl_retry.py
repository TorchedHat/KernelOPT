import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=2),
        triton.Config({'XBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_55(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # 2D grid: axis 0 = channel tiles (C=256), axis 1 = spatial positions (S=1960)
    pid_c = tl.program_id(0)  # channel tile index
    pid_s = tl.program_id(1)  # spatial index (one block per spatial position)
    c_off = pid_c * XBLOCK + tl.arange(0, XBLOCK)[:]
    c_mask = c_off < 256
    s_idx = pid_s  # scalar spatial position in [0, 1960)
    # Input is channel-last contiguous with stride-256 in channel dim
    in_idx = s_idx * 256 + c_off
    tmp0 = tl.load(in_ptr0 + in_idx, c_mask)
    tmp1 = tl.load(in_ptr1 + c_off, c_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Output has stride-512 in channel dim
    out_idx = c_off + 512 * s_idx
    tl.store(out_ptr0 + out_idx, tmp2, c_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 501760
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_55[grid](arg_0, arg_1, arg_2, 501760)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 256, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
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
