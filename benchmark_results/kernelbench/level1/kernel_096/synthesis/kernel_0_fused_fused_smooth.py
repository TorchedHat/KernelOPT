import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr, tgt_ptr, partial_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tgt = tl.load(tgt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = tl.abs(pred - tgt)
    loss = tl.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    partial_sum = tl.sum(loss, axis=0)
    tl.store(partial_ptr + pid, partial_sum)