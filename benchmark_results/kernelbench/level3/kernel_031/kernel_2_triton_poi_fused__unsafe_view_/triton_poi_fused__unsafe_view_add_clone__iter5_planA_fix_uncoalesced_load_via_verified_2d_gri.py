import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: pid(0) = x1 in [0, 16384), pid(1) = col-block index
    # XBLOCK is used as CBLOCK (col-block size)
    x1_numel = 16384
    col_numel = 256

    pid_x = tl.program_id(0)   # x1 in [0, 16384)
    pid_y = tl.program_id(1)   # col-block index

    col_base = pid_y * XBLOCK
    col_off = tl.arange(0, XBLOCK)
    col = col_base + col_off   # col = x0 + 32*x2, range [0, 256)
    col_mask = col < col_numel

    x1 = pid_x

    # ALGEBRAIC PROOF:
    # original = 384*(col//128) + 768*x1 + 768*(col//256) + (col%128)
    # col < 256 => col//256 = 0 always
    # => 768*x1 + 384*(col//128) + (col%128)
    # => (2*x1 + col//128) * 384 + (col%128)
    in_row = 2 * x1 + col // 128
    in_col = col % 128
    tmp0 = tl.load(in_ptr0 + in_row * 384 + in_col, mask=col_mask)

    # in_ptr1: original = (x0+32*x2)%128 = col%128 = in_col
    tmp1 = tl.load(in_ptr1 + in_col, mask=col_mask, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.1767766952966369, tl.float32)
    result = tmp2 * tmp3

    # out_ptr0: x2=col//32, x0=col%32
    # original x3 = x2*524288 + x1*32 + x0
    x2_out = col // 32
    x0_out = col % 32
    out_idx = x2_out * 524288 + x1 * 32 + x0_out
    tl.store(out_ptr0 + out_idx, result, mask=col_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2[grid](arg_0, arg_1, arg_2, 4194304)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
