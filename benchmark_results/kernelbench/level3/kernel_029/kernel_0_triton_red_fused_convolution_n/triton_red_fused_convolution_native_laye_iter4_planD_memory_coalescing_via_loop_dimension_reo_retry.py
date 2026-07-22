import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 256}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 256}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 96, 'R0_BLOCK': 64},  num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    # xindex covers (batch, channel) = 10*96 = 960
    # r0 covers spatial = 56*56 = 3136
    # This makes the dominant load stride-1 contiguous in r0
    xnumel = 960
    r0_numel = 3136
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    # xindex encodes (batch, channel): x_channel = xindex % 96, x_batch = xindex // 96
    x_channel = (xindex % 96)
    x_batch = xindex // 96
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    # Load channel-level bias (in_ptr1) once outside loop - indexed by x_channel
    tmp1_bias = tl.load(in_ptr1 + x_channel, mask=xmask, eviction_policy='evict_last', other=0.0)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_spatial = r0_index
        # Load: in_ptr0 has shape (10, 96, 56, 56) strides (301056, 3136, 56, 1)
        # Address = r0_spatial + 3136*x_channel + 301056*x_batch
        # r0_spatial varies with stride=1 -> contiguous access!
        _off0 = r0_spatial + 3136*x_channel + 301056*x_batch
        tmp0 = tl.load(in_ptr0 + _off0, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1_bias
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    # Load LN weight and bias (indexed by channel, broadcast over spatial)
    tmp20_weight = tl.load(in_ptr2 + x_channel, mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp22_lnbias = tl.load(in_ptr3 + x_channel, mask=xmask, eviction_policy='evict_last', other=0.0)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_spatial = r0_index
        _off0b = r0_spatial + 3136*x_channel + 301056*x_batch
        tmp10 = tl.load(in_ptr0 + _off0b, mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 + tmp1_bias
        tmp13 = tmp12 - tmp4
        tmp14 = tl.full([1, 1], 3136.0, tl.float32)
        tmp15 = (tmp8 / tmp14)
        tmp16 = tl.full([1, 1], 1e-05, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20_weight
        tmp23 = tmp21 + tmp22_lnbias
        # Output: out_ptr2 has shape (10, 3136, 96) strides (301056, 96, 1)
        # Address = x_channel + 96*r0_spatial + 301056*x_batch
        tl.store(out_ptr2 + (x_channel + 96*r0_spatial + 301056*x_batch), tmp23, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 31360
    r0_numel = 96
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 96, 56, 56), (301056, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
