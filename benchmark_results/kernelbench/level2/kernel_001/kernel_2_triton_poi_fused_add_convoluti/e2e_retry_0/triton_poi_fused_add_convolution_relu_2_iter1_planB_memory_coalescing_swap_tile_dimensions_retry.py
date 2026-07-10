import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_relu_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Swapped: xnumel = batch*channel (stride-1 axis), ynumel = spatial
    xnumel = 16384   # batch * channel = 128 * 128
    ynumel = 15876   # spatial H*W = 126*126

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    xmask = xindex < xnumel  # always True since 16384 divisible by XBLOCK configs

    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < ynumel  # needed since 15876 is not power of 2

    x0 = xindex % 128   # channel — stride=1 in in_ptr0 => COALESCED
    x1 = xindex // 128  # batch
    y2 = yindex          # spatial index (h*126 + w)

    # Load from in_ptr0: channel + 128*spatial + 2032128*batch
    # x0 shape [1,XBLOCK], y2 shape [YBLOCK,1], x1 shape [1,XBLOCK]
    # broadcast => [YBLOCK, XBLOCK]
    tmp0 = tl.load(in_ptr0 + (x0 + 128*y2 + 2032128*x1), xmask & ymask, eviction_policy='evict_last')
    # in_ptr1: bias, shape (128,), indexed by channel x0
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    # in_ptr2: residual, shape (128,1,1) strides (1,1,1), indexed by channel x0
    tmp5 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tmp6 = tmp4 + tmp5
    # Store to out_ptr0: spatial + 15876*(channel + 128*batch) = y2 + 15876*xindex
    tl.store(out_ptr0 + (y2 + 15876*xindex), tmp6, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 16384
    xnumel = 15876
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_convolution_relu_2[grid](arg_0, arg_1, arg_2, arg_3, 16384, 15876)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 128, 126, 126), (2032128, 1, 16128, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 126, 126), (2032128, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
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
