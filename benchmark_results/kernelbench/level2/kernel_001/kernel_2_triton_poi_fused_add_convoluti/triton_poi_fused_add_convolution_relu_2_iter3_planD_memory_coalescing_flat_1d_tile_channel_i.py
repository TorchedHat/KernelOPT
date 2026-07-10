import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_relu_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D tile: numel = ynumel * xnumel = 16384 * 15876 = 260046848
    # YBLOCK is always 1 in all configs; we use XBLOCK as the flat block size
    numel = 16384 * 15876
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    # Decompose flat index: channel is innermost (stride-1 in in_ptr0) -> COALESCED loads
    channel = offsets % 128                   # y0: stride-1 in in_ptr0
    tmp_sb  = offsets // 128
    spatial  = tmp_sb % 15876                 # x2: stride-1 in out_ptr0
    batch    = tmp_sb // 15876                # y1: batch index [0,127]
    # Load: address = channel + 128*spatial + 2032128*batch (identical to y0+128*x2+2032128*y1)
    tmp0 = tl.load(in_ptr0 + (channel + 128*spatial + 2032128*batch), mask=mask)
    # Bias and residual indexed by channel
    tmp1 = tl.load(in_ptr1 + channel, mask=mask)
    tmp5 = tl.load(in_ptr2 + channel, mask=mask)
    # Compute relu(x+bias)+residual
    tmp2 = tmp0 + tmp1
    tmp4 = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp2)
    tmp6 = tmp4 + tmp5
    # Store: address = spatial + 15876*(batch*128+channel) (identical to x2+15876*y3)
    y3 = batch * 128 + channel
    tl.store(out_ptr0 + (spatial + 15876*y3), tmp6, mask=mask)

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
