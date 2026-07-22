import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2),
    triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4),
    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # numel = 46080 = 32 * 160 * 3 * 3
    # Output strides: (1440, 1, 480, 160) — channel is stride-1 (fastest dim)
    # Input  strides: (1440, 9,  3,  1)
    # Use XBLOCK for 1D flat iteration; YBLOCK=1 always
    numel = 46080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < numel
    # Decompose flat output index into (b, c, h, w) using output strides (1440,1,480,160)
    b    = xindex // 1440
    rem  = xindex % 1440
    h    = rem // 480
    rem2 = rem % 480
    w    = rem2 // 160
    c    = rem2 % 160
    # Compute input index using input strides (1440, 9, 3, 1)
    in_idx = b * 1440 + c * 9 + h * 3 + w
    tmp0 = tl.load(in_ptr0 + in_idx, xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + xindex, tmp0, xmask)  # stride-1, fully coalesced

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 5120
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, 5120, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
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
