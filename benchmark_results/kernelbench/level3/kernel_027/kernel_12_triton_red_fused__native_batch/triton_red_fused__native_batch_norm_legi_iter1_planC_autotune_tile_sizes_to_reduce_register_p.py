import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 16},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14336
    r0_numel = 112
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (14336 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (14592 + x0 + 512*((r0_2 % 28)) + 28672*(r0_2 // 28) + 114688*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.maximum(tmp0, tmp1)
        tmp4 = tl.maximum(tmp2, tmp3)
        tmp6 = tl.maximum(tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 14336
    r0_numel = 112
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_12[grid](arg_0, arg_1, 14336, 112)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 256, 7), (1792, 1, 256), device='cuda:0', dtype=torch.float32)
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
