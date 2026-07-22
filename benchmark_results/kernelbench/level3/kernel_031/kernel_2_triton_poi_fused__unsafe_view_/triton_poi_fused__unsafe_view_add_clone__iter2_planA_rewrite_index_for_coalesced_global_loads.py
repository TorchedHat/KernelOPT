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
        triton.Config({'XBLOCK': 128}, num_warps=8),
    ],
    key=['num_rows', 'num_cols'],
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: pid(1) = in_ptr0 row [0, 32768), pid(0) = column tile [0, cdiv(128, XBLOCK))
    # in_ptr0 shape: (32768, 384), but only columns [0..127] are accessed per row
    # in_ptr0 row = 2*x1 + col//128, in_ptr0 col = col%128
    # where col = x0 + 32*x2, x0 in [0,31], x1 in [0,16383], x2 in [0,7]
    # Inverse: given pid_row, col_offset:
    #   x1 = pid_row // 2
    #   col = (pid_row % 2) * 128 + col_offset   (col in [0,255])
    #   x2 = col // 32
    #   x0 = col % 32
    # Output index x3 = x2*524288 + x1*32 + x0
    num_cols = 128
    pid_row = tl.program_id(1)   # which row of in_ptr0 [0, 32768)
    pid_col = tl.program_id(0)   # which column tile
    col_offset = pid_col * XBLOCK + tl.arange(0, XBLOCK)
    col_mask = col_offset < num_cols  # guard for 128 % XBLOCK != 0

    # COALESCED LOAD: all threads load consecutive cols from same row of in_ptr0
    tmp0 = tl.load(in_ptr0 + pid_row * 384 + col_offset, mask=col_mask, other=0.0)

    # Bias: in_ptr1 indexed by col_offset (same as in_ptr0 column)
    tmp1 = tl.load(in_ptr1 + col_offset, mask=col_mask, other=0.0, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.1767766952966369, tl.float32)
    tmp4 = tmp2 * tmp3

    # Compute output index:
    # col (in original sense) = (pid_row % 2) * 128 + col_offset
    # x1 = pid_row // 2
    # x2 = col // 32
    # x0 = col % 32
    # x3 = x2 * 524288 + x1 * 32 + x0
    x1 = pid_row // 2
    col = (pid_row % 2) * 128 + col_offset   # shape [XBLOCK]
    x2 = col // 32
    x0 = col % 32
    x3 = x2 * 524288 + x1 * 32 + x0

    tl.store(out_ptr0 + x3, tmp4, mask=col_mask)

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
