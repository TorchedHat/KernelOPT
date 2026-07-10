import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16},  num_warps=4),
        triton.Config({'XBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK

    # Grid is (cdiv(16384, XBLOCK) * 16,)
    # Each CTA handles one batch row: x1_scalar = batch index, x0_scalar = spatial start
    pid = tl.program_id(0)
    tiles_per_batch: tl.constexpr = 16384 // XBLOCK
    x1_scalar = pid // tiles_per_batch   # batch index [0..15]
    x0_scalar = (pid % tiles_per_batch) * XBLOCK  # spatial start [0..16383]

    # 2D block pointer: shape=(64 channels, 16384 spatial), strides=(16384, 1)
    # order=(1,0): fast axis = dim-1 = spatial (stride=1) -> COALESCED
    load_ptr = tl.make_block_ptr(
        base=in_out_ptr0 + x1_scalar * 1048576,
        shape=(64, 16384),
        strides=(16384, 1),
        offsets=(0, x0_scalar),
        block_shape=(R0_BLOCK, XBLOCK),
        order=(1, 0)
    )
    tmp0_T = tl.load(load_ptr, boundary_check=(0, 1), padding_option='zero')  # [64, XBLOCK]
    tmp0 = tl.trans(tmp0_T)  # [XBLOCK, 64] for reduction over channels

    # Bias: in_ptr0 shape (1,64,1,1,1) stride (64,1,...) -> index by channel
    tmp3 = tl.load(in_ptr0 + tl.arange(0, R0_BLOCK)[None, :])  # [1, 64] broadcast to [XBLOCK, 64]

    tmp1 = tl.full([1, 1], 32.0, tl.float32)
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3  # [XBLOCK, 64]

    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = tl.max(tmp7, 1)[:, None].to(tl.float32)
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp17 = tmp11 / tmp14
    tmp18 = libdevice.tanh(tmp17)
    tmp19 = tl.full([1, 1], 2.0, tl.float32)
    tmp20 = tmp18 * tmp19  # [XBLOCK, 64]

    # Store back: transpose [XBLOCK,64] -> [64,XBLOCK] for coalesced write
    store_ptr = tl.make_block_ptr(
        base=in_out_ptr0 + x1_scalar * 1048576,
        shape=(64, 16384),
        strides=(16384, 1),
        offsets=(0, x0_scalar),
        block_shape=(R0_BLOCK, XBLOCK),
        order=(1, 0)
    )
    tl.store(store_ptr, tl.trans(tmp20), boundary_check=(0, 1))

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
