import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: axis-0 = channel tiles (CBLOCK=256, masked to C=208), axis-1 = spatial tiles (SBLOCK=4)
    CBLOCK: tl.constexpr = 256   # must be power of 2; actual channels C=208
    SBLOCK: tl.constexpr = 4
    C: tl.constexpr = 208
    S: tl.constexpr = 501760     # 10 * 224 * 224
    pid_c = tl.program_id(0)     # always 0 since ceil(208/256)=1
    pid_s = tl.program_id(1)     # range [0, cdiv(S, SBLOCK))
    c_off = pid_c * CBLOCK + tl.arange(0, CBLOCK)  # [0..255]
    c_mask = c_off < C           # guard: only first 208 channels are valid
    s_base = pid_s * SBLOCK
    # Load bias once per CTA — 208 floats, fits in L1
    bias = tl.load(in_ptr1 + c_off, mask=c_mask, other=0.0, eviction_policy='evict_last')
    # Loop over SBLOCK spatial positions (unrolled by compiler via static_range)
    for i in tl.static_range(SBLOCK):
        s_idx = s_base + i
        valid_s = s_idx < S
        # in_ptr0: NHWC flat, channel stride=208 => index = s_idx*208 + c_off
        vals = tl.load(in_ptr0 + s_idx * C + c_off, mask=c_mask & valid_s, other=0.0)
        result = vals + bias
        # out_ptr0: padded channel stride=512 => index = s_idx*512 + c_off
        tl.store(out_ptr0 + s_idx * 512 + c_off, result, mask=c_mask & valid_s)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 104366080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_10[grid](arg_0, arg_1, arg_2, 104366080, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 208, 224, 224), (10436608, 1, 46592, 208), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((208,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 208, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
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
