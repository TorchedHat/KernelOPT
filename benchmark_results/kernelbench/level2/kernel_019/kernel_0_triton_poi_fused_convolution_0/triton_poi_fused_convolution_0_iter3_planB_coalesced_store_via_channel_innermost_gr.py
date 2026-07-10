import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=8, num_stages=1),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Total flat output elements: numel = 10 * 3 * 224 * 224 = 1505280
    # Use program_id(0) to tile over flat output index for coalesced stores
    numel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    # Consecutive threads -> consecutive out_flat -> coalesced store
    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), 16), 16)
    out_flat = xoffset + xarange
    mask = out_flat < numel

    # Decompose flat output index into (n, c, hw) for NHWC output
    # NHWC strides: (150528, 1, 672, 3)
    # out_flat = c + 3*hw + 150528*n
    # => c  = out_flat % 3
    # => hw = (out_flat // 3) % 50176
    # => n  = out_flat // 150528
    c  = out_flat % 3
    hw = (out_flat // 3) % 50176
    n  = out_flat // 150528

    # Input is NCHW with strides (150528, 50176, 224, 1)
    # in_flat = n*150528 + c*50176 + hw
    in_flat = hw + 50176 * c + 150528 * n

    tmp0 = tl.load(in_ptr0 + in_flat, mask, eviction_policy='evict_first')
    # Store to consecutive out_flat addresses -> perfectly coalesced (stride 1)
    tl.store(out_ptr0 + out_flat, tmp0, mask)

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
