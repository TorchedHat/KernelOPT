import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # 2D grid: pid_x over x1 (16384 rows), pid_y over col (256 cols)
    # XBLOCK covers x1 rows, YBLOCK=32 hardcoded covers col dimension
    YBLOCK: tl.constexpr = 32
    pid_x = tl.program_id(0)  # over x1 dimension (16384 rows)
    pid_y = tl.program_id(1)  # over col dimension (256 cols)
    x1_offs = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    col_offs = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    x1 = x1_offs[:, None]    # [XBLOCK, 1]
    col = col_offs[None, :]  # [1, YBLOCK]
    x0 = col % 32            # [1, YBLOCK]
    x2 = col // 32           # [1, YBLOCK]
    x1_mask = x1_offs[:, None] < 16384
    col_mask = col_offs[None, :] < 256
    mask = x1_mask & col_mask
    in0_col = 384 * (col // 128) + (col % 128)  # [1, YBLOCK] coalesced
    in0_idx = x1 * 384 + in0_col               # [XBLOCK, YBLOCK]
    tmp0 = tl.load(in_ptr0 + in0_idx, mask)
    in1_idx = col % 128                         # [1, YBLOCK]
    tmp1 = tl.load(in_ptr1 + in1_idx, col_mask, eviction_policy='evict_last')
    tmp4 = (tmp0 + tmp1) * 0.1767766952966369
    out_idx = x2 * 524288 + x1 * 32 + x0       # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + out_idx, tmp4, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2[grid](arg_0, arg_1, arg_2, 4194304, XBLOCK=128)
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
