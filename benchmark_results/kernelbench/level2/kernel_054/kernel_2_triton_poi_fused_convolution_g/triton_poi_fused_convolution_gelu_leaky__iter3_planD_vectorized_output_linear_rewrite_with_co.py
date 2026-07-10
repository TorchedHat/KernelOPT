import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 4096, 'XBLOCK': 1}, num_warps=8, num_stages=4),
    ],
    key=['ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D iteration over output tensor's natural linear layout
    # Output shape (64,64,254,254) strides (4129024,64516,254,1) -> w is stride-1
    # Input  shape (64,64,254,254) strides (4129024,1,16256,64)
    # ynumel = total numel = 264257536
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < ynumel
    # Decompose flat output index into (batch, channel, h, w)
    w       = offsets % 254
    tmp_hw  = offsets // 254
    h       = tmp_hw % 254
    tmp_chw = tmp_hw // 254
    channel = tmp_chw % 64
    batch   = tmp_chw // 64
    # Compute input flat index using input strides (4129024, 1, 16256, 64)
    in_flat = batch * 4129024 + channel + h * 16256 + w * 64
    # Load input, bias, scale
    x     = tl.load(in_ptr0 + in_flat, mask=mask)
    bias  = tl.load(in_ptr1 + channel, mask=mask, eviction_policy='evict_last')
    scale = tl.load(in_ptr2 + channel, mask=mask, eviction_policy='evict_last')
    # Compute: (x + bias) * scale -> leaky_relu -> gelu
    val = (x + bias) * scale
    val_leaky = tl.where(val > 0.0, val, val * 0.01)
    val_gelu  = val_leaky * 0.5 * (1.0 + libdevice.erf(val_leaky * 0.7071067811865476))
    # Store to output (flat index = offsets, stride-1 in w -> coalesced vectorized)
    tl.store(out_ptr0 + offsets, val_gelu, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 4129024
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_gelu_leaky_relu_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 4129024, 64)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((64, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 64, 254, 254), (4129024, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
