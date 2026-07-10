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
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK now covers ynumel=16384 (channel*batch, stride=1 in in_ptr0) -> COALESCED
    # YBLOCK now covers xnumel=15876 (spatial, stride=128 in in_ptr0)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK] channel*batch
    xmask = xindex < ynumel  # channel*batch boundary mask
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1] spatial
    ymask = yindex < xnumel  # spatial boundary mask (15876 is not a power of 2)
    full_mask = ymask & xmask

    c  = xindex % 128   # channel index 0..127, stride=1 in in_ptr0
    n  = xindex // 128  # batch index 0..127
    hw = yindex         # spatial index 0..15875

    # in_ptr0 stride: (2032128, 1, 16128, 128) -> address = n*2032128 + c*1 + hw*128
    # Consecutive threads differ by c (stride=1) -> COALESCED
    tmp0 = tl.load(in_ptr0 + (c + 128*hw + 2032128*n), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + c, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + c, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tl.minimum(tmp2, tmp3)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 2.0, tl.float32)
    tmp8 = tmp6 * tmp7
    # out_ptr0 stride: (2032128, 15876, 126, 1) -> address = n*2032128 + c*15876 + hw
    tl.store(out_ptr0 + (hw + 15876*c + 2032128*n), tmp8, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 16384
    xnumel = 15876
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 16384, 15876)
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
