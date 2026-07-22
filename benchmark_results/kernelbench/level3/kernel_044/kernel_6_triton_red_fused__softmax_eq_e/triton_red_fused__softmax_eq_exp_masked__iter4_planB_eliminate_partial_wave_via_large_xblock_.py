import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # XBLOCK=1: grid=524288, coalesced access
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=1),
        # XBLOCK=2
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        # XBLOCK=4
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        # XBLOCK=8
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        # XBLOCK=16
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        # XBLOCK=32
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        # XBLOCK=64
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        # XBLOCK=128: grid=4096 (1.94 waves)
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
        # XBLOCK=256: grid=2048 (one full wave, eliminates partial wave)
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=16, num_stages=2),
        # XBLOCK=512: grid=1024 (sub-wave, all SMs busy throughout)
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 512}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 512}, num_warps=16, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    x0 = (xindex % 512)
    x3 = xindex
    # --- SINGLE LOAD of both inputs ---
    tmp0 = tl.load(in_ptr0 + (r0_index + 1024*x0), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r0_index + 512*x3), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
    # --- Apply mask and scale ---
    tmp1 = tl.full([1, 1], 0.0, tl.float32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp7 = tl.where(tmp2, tmp6, tmp5)
    # --- Compute softmax in registers (single pass) ---
    row_max = tl.max(tmp7, axis=1, keep_dims=True)    # [XBLOCK, 1]
    exp_val = tl.exp(tmp7 - row_max)                  # [XBLOCK, R0_BLOCK]
    row_sum = tl.sum(exp_val, axis=1, keep_dims=True)  # [XBLOCK, 1]
    out = exp_val / row_sum                            # [XBLOCK, R0_BLOCK]
    # --- SINGLE STORE ---
    tl.store(in_out_ptr0 + (r0_index + 512*x3), out, mask=xmask & r0_mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 524288
    r0_numel = 512
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 512), (2097152, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
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
