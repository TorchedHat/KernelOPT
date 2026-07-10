import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clamp_convolution_div_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    EPT: tl.constexpr = 8
    block_start = tl.program_id(0) * (XBLOCK * EPT)
    lane = tl.arange(0, XBLOCK)[:]
    clamp_lo = tl.full([1], -1.0, tl.float32)
    scale    = tl.full([1],  0.5, tl.float32)
    for i in tl.static_range(EPT):
        xindex = block_start + i * XBLOCK + lane
        xmask  = xindex < xnumel
        x1 = ((xindex // 424175) % 128)
        tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
        tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tl.maximum(tmp2, clamp_lo)
        tmp6 = tmp4 * scale
        tl.store(in_out_ptr0 + xindex, tmp6, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 868710400
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clamp_convolution_div_0[grid](arg_0, arg_1, 868710400, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 128, 47, 95, 95), (54294400, 424175, 9025, 95, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
