import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 56)
    x2 = xindex // 7168
    x3 = xindex
    # Factor common address to reduce redundant integer arithmetic
    base = x0 + 256*x1 + 28672*x2
    # Issue ALL 4 loads consecutively before any computation
    tmp0 = tl.load(in_ptr0 + base,           mask=xmask)  # row0, col0
    tmp1 = tl.load(in_ptr0 + (base + 128),   mask=xmask)  # row0, col1
    tmp3 = tl.load(in_ptr0 + (base + 14336), mask=xmask)  # row1, col0
    tmp5 = tl.load(in_ptr0 + (base + 14464), mask=xmask)  # row1, col1
    # All ALU after all loads: no L1TEX stall between max ops
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + x3, tmp6, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3211264
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8[grid](arg_0, arg_1, 3211264)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 128, 112, 112), (1605632, 1, 14336, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
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
