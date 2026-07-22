import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},   num_warps=1),
        triton.Config({'XBLOCK': 2},   num_warps=1),
        triton.Config({'XBLOCK': 4},   num_warps=1),
        triton.Config({'XBLOCK': 8},   num_warps=1),
        triton.Config({'XBLOCK': 16},  num_warps=1),
        triton.Config({'XBLOCK': 32},  num_warps=1),
        triton.Config({'XBLOCK': 64},  num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=4),
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    xnumel = 192
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9[grid](arg_0, 192)
    return arg_0

def get_inputs():
    arg_0 = rand_strided((2, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    return [arg_0]

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
