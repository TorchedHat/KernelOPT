import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4},  num_warps=2),
        triton.Config({'XBLOCK': 8},  num_warps=2),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # XBLOCK here is repurposed as YBLOCK (spatial rows per block)
    # Channel count is fixed at 64 (C=64, innermost dimension in NHWC layout)
    C: tl.constexpr = 64
    ynumel = 31360  # = 2007040 // 64 = N*H*W spatial positions
    pid_y = tl.program_id(0)  # spatial block
    # Channel indices: shape [C] = [64]
    x0 = tl.arange(0, C)
    # Load channel params ONCE per block (hoisted out of spatial loop)
    mean   = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
    var    = tl.load(in_ptr1 + x0, eviction_policy='evict_last')
    weight = tl.load(in_ptr2 + x0, eviction_policy='evict_last')
    bias   = tl.load(in_ptr3 + x0, eviction_policy='evict_last')
    # Pre-compute rsqrt ONCE per channel per block (hoisted out of spatial loop)
    inv_std = tl.math.rsqrt(var + 1e-05)
    y_start = pid_y * XBLOCK
    # Loop over XBLOCK spatial rows (XBLOCK is autotuned as YBLOCK)
    for yi in tl.static_range(XBLOCK):
        y = y_start + yi
        ymask = y < ynumel
        idx = y * C + x0
        x_val = tl.load(in_out_ptr0 + idx, mask=ymask, other=0.0, eviction_policy='evict_first')
        normed = (x_val - mean) * inv_std
        out = tl.math.fma(normed, weight, bias)
        out = tl.maximum(0.0, out)
        tl.store(in_out_ptr0 + idx, out, mask=ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2007040
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
