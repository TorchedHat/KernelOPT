import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16384}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 8192}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    # New layout:
    #   XBLOCK=128 covers r0_2 (channel dim, size 128) — slow dim, axis=0
    #   R0_BLOCK covers x0 (spatial dim, size 16641) — fast dim, axis=1 (stride-1 in output)
    # Grid: 128 programs (one per batch element x1)
    # Each program handles all 128 channels x a chunk of spatial positions
    xnumel = 2130048
    r0_numel = 128

    # pid encodes batch index (x1)
    x1_pid = tl.program_id(0)

    # r0_tile covers r0_2 (channel dimension, size 128) — [XBLOCK, 1]
    r0_tile = tl.arange(0, XBLOCK)[:, None]   # [128, 1]
    r0_mask = r0_tile < 128

    # Load bias vectors (channel-indexed, broadcast over spatial)
    tmp1 = tl.load(in_ptr1 + r0_tile, mask=r0_mask, other=0.0)   # [XBLOCK, 1]
    tmp12 = tl.load(in_ptr2 + r0_tile, mask=r0_mask, other=0.0)  # [XBLOCK, 1]

    # Loop over spatial positions in R0_BLOCK chunks
    x0_base = tl.arange(0, R0_BLOCK)[None, :]  # [1, R0_BLOCK]

    for x0_offset in tl.range(0, 16641, R0_BLOCK):
        x_tile = x0_offset + x0_base   # [1, R0_BLOCK] — spatial positions
        x_mask = x_tile < 16641        # [1, R0_BLOCK]

        combined_mask = r0_mask & x_mask  # [XBLOCK, R0_BLOCK]

        # Load input: in_ptr0 strides (2130048, 1, 16512, 128)
        # Address: r0_tile (channel, stride=1) + 128*x_tile (spatial, stride=128) + 2130048*x1_pid (batch)
        tmp0 = tl.load(in_ptr0 + (r0_tile + 128 * x_tile + 2130048 * x1_pid),
                       mask=combined_mask, other=0.0)  # [XBLOCK, R0_BLOCK]

        tmp2 = tmp0 + tmp1  # [XBLOCK, R0_BLOCK] — add channel bias

        # Softmax reduction over channel dim (axis=0, the XBLOCK dim)
        # For each spatial position, reduce over 128 channels
        tmp_max = tl.max(tmp2, axis=0, keep_dims=True)               # [1, R0_BLOCK]
        tmp_exp = tl.exp(tmp2 - tmp_max)                              # [XBLOCK, R0_BLOCK]
        tmp_sum = tl.sum(tmp_exp, axis=0, keep_dims=True)             # [1, R0_BLOCK]
        tmp_softmax = tmp_exp / tmp_sum                               # [XBLOCK, R0_BLOCK]

        # Epilogue: sigmoid(2 * (softmax + bias2))
        tmp13 = tmp_softmax + tmp12                                   # [XBLOCK, R0_BLOCK]
        tmp15 = tmp13 * 2.0
        tmp16 = tl.sigmoid(tmp15)

        # Store: out_ptr2 strides (2130048, 16641, 129, 1)
        # Address: x_tile (spatial, stride=1) + 16641*r0_tile (channel, stride=16641) + 2130048*x1_pid (batch)
        # x_tile is the fast-varying warp dimension (stride-1) -> COALESCED
        tl.store(out_ptr2 + (x_tile + 16641 * r0_tile + 2130048 * x1_pid),
                 tmp16, mask=combined_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 2130048
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2[grid](arg_0, arg_1, arg_2, arg_3, 2130048, 128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 128, 129, 129), (2130048, 1, 16512, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 129, 129), (2130048, 16641, 129, 1), device='cuda:0', dtype=torch.float32)
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
