import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0_out = xindex % 512          # padded channel index (0..511)
    x1_out = xindex // 512         # spatial index (N*H*W)
    xmask = x0_out < 48            # only valid channels
    x_in = x0_out + 48 * x1_out   # flat input index (NHWC, 48 channels)
    tmp0 = tl.load(in_ptr0 + x_in, xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0_out, xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + xindex, tmp2, xmask)  # COALESCED: contiguous output addresses

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 24084480
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_9[grid](arg_0, arg_1, arg_2, 24084480)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 48, 224, 224), (2408448, 1, 10752, 48), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((48,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 48, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
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
