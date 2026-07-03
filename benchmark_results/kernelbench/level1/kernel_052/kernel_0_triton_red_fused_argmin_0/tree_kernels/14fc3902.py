import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_argmin_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524160
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 4095)
    x1 = xindex // 4095
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("inf"), tl.float32)
    _tmp2_index = tl.full([XBLOCK, R0_BLOCK], 2147483647, tl.int32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        load_idx = x0 + 4095 * r0_index + 16773120 * x1
        tmp0 = tl.load(in_ptr0 + load_idx, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        _tmp2_next, _tmp2_index_next = triton_helpers.minimum_with_index(
            _tmp2, _tmp2_index, tmp1, rindex
        )
        _tmp2 = tl.where(r0_mask & xmask, _tmp2_next, _tmp2)
        _tmp2_index = tl.where(r0_mask & xmask, _tmp2_index_next, _tmp2_index)
    tmp2_val, tmp2_idx = triton_helpers.min_with_index(_tmp2, _tmp2_index, 1)
    tmp2 = tmp2_idx[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 524160
    r0_numel = 4096
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_argmin_0[grid](arg_0, arg_1, 524160, 4096)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 4096, 4095), (16773120, 4095, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 4095), (4095, 1), device='cuda:0', dtype=torch.int64)
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
