import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 8}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8}, num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_col = r0_index
    x_batch   = xindex // 8192
    x_channel = (xindex % 8192) // 128
    x_row     = xindex % 128
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_col + 128*x_row + 16384*x_channel + 1048576*x_batch), xmask, eviction_policy='evict_first', other=0.0)
    tmp3 = tl.load(in_ptr0 + (x_channel), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * 0.03125
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = triton_helpers.max2(tmp7, 1)[:, None].to(tl.float32)
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    # Reuse tmp11 (exp_shifted) instead of recomputing tl.exp(tmp4 - tmp9)
    tmp17 = (tmp11 / tmp14)
    tmp18 = libdevice.tanh(tmp17)
    tmp20 = tmp18 + tmp18
    tl.store(in_out_ptr0 + (r0_col + 128*x_row + 16384*x_channel + 1048576*x_batch), tmp20, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 262144
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1[grid](arg_0, arg_1, 262144, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 1, 128, 128), (1048576, 16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
