import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 8192}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 8192}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=2, num_stages=1),
    triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
], key=['r0_numel'])
@triton.jit
def triton_red_fused_smooth_l1_loss_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    r0_numel = 8192
    r0_base = tl.arange(0, R0_BLOCK)
    _acc = tl.zeros([R0_BLOCK], dtype=tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        tmp0 = tl.load(in_ptr0 + r0_index, mask=r0_mask, eviction_policy='evict_first', other=0.0)
        _acc = tl.where(r0_mask, _acc + tmp0, _acc)
    tmp2 = tl.sum(_acc, axis=0)
    tmp5 = tmp2 / 1073741824.0
    tl.debug_barrier()
    tl.store(in_out_ptr0, tmp5)


@triton.jit

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
