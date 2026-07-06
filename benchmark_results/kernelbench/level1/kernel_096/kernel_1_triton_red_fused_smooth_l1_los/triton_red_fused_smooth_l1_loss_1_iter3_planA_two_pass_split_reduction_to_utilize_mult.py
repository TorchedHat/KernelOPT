import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'R0_BLOCK': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'R0_BLOCK': 1024, 'XBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'R0_BLOCK': 512,  'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'R0_BLOCK': 512,  'XBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'R0_BLOCK': 256,  'XBLOCK': 1}, num_warps=4, num_stages=4),
        triton.Config({'R0_BLOCK': 256,  'XBLOCK': 1}, num_warps=8, num_stages=4),
    ],
    key=['r0_numel'],
    reset_to_zero=['in_out_ptr0'],
)
@triton.jit
def triton_red_fused_smooth_l1_loss_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    # Split the 8192-element reduction across N_SPLITS=8 blocks
    # Each block handles ELEMS_PER_SPLIT=1024 elements
    split_id = tl.program_id(0)
    elems_per_split = r0_numel // 8
    base_offset = split_id * elems_per_split
    r0_base = tl.arange(0, R0_BLOCK)
    _acc = tl.zeros([R0_BLOCK], dtype=tl.float32)
    for r0_offset in tl.range(0, elems_per_split, R0_BLOCK):
        r0_index = base_offset + r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        tmp0 = tl.load(in_ptr0 + r0_index, mask=r0_mask, eviction_policy='evict_last', other=0.0)
        _acc = tl.where(r0_mask, _acc + tmp0, _acc)
    partial_sum = tl.sum(_acc, axis=0)
    partial_sum = partial_sum / 1073741824.0
    tl.atomic_add(in_out_ptr0, partial_sum)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1
    r0_numel = 8192
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_smooth_l1_loss_1[grid](arg_0, arg_1, 1, 8192)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
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
