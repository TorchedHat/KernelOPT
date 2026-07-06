import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=1),
        triton.Config({'XBLOCK': 1},  num_warps=2),
        triton.Config({'XBLOCK': 2},  num_warps=1),
        triton.Config({'XBLOCK': 2},  num_warps=2),
        triton.Config({'XBLOCK': 4},  num_warps=1),
        triton.Config({'XBLOCK': 4},  num_warps=2),
        triton.Config({'XBLOCK': 4},  num_warps=4),
        triton.Config({'XBLOCK': 8},  num_warps=2),
        triton.Config({'XBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=2),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=4),
    ],
    key=[],  # xnumel and r0_numel are compile-time constants
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 16
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_native_layer_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 16, 32)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((16, 1, 1, 1, 32), (32, 512, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 1, 32), (32, 512, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 1, 1, 1, 32), (32, 512, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 1, 1, 1), (1, 16, 16, 16), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16, 1, 1, 1), (1, 16, 16, 16), device='cuda:0', dtype=torch.float32)
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
