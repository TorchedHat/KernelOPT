import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr):
    # 2D column-major tiling: XBLOCK = XCOLS (column tile size)
    # Grid is (8192 // XBLOCK, 1024 // 8) = (ncol_tiles, 128)
    # pid(0) = column tile, pid(1) = row tile (8 rows per tile)
    pid_col = tl.program_id(0)  # column tile index
    pid_row = tl.program_id(1)  # row tile index

    col_offsets = pid_col * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    row_offsets = pid_row * 8 + tl.arange(0, 8)            # [8]

    # Load 6 broadcast vectors ONCE per CTA (column-indexed only)
    # No mask needed: XBLOCK divides 8192 exactly for all valid configs
    tmp0  = tl.load(in_ptr0 + col_offsets)   # addmm bias [XBLOCK]
    tmp3  = tl.load(in_ptr1 + col_offsets)   # mul_factor [XBLOCK]
    tmp5  = tl.load(in_ptr2 + col_offsets)   # running_mean [XBLOCK]
    tmp7  = tl.load(in_ptr3 + col_offsets)   # running_var [XBLOCK]
    tmp16 = tl.load(in_ptr4 + col_offsets)   # gamma [XBLOCK]
    tmp18 = tl.load(in_ptr5 + col_offsets)   # beta [XBLOCK]

    # Precompute rsqrt once per column, reuse across rows
    tmp8  = tl.full([XBLOCK], 1e-05, tl.float32)
    tmp9  = tmp7 + tmp8
    tmp14 = libdevice.rsqrt(tmp9)  # [XBLOCK]

    # 2D flat index: [8, XBLOCK]
    flat_idx = row_offsets[:, None] * 8192 + col_offsets[None, :]

    # Load per-element addmm output (read-modify-write)
    tmp1 = tl.load(in_out_ptr0 + flat_idx)  # [8, XBLOCK]

    # Compute BN with broadcast from [XBLOCK] to [8, XBLOCK]
    tmp2  = tmp0[None, :] + tmp1
    tmp4  = tmp2  * tmp3[None, :]
    tmp6  = tmp4  - tmp5[None, :]
    tmp15 = tmp6  * tmp14[None, :]
    tmp17 = tmp15 * tmp16[None, :]
    tmp19 = tmp17 + tmp18[None, :]

    tl.store(in_out_ptr0 + flat_idx, tmp19)  # no mask needed

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
