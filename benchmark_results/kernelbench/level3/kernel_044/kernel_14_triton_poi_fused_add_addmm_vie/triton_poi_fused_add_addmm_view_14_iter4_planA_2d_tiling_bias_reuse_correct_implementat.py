import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_addmm_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: each CTA processes YBLOCK=8 rows x 1024 columns (masked to 768)
    # Bias vectors (in_ptr1, in_ptr2) loaded once per CTA and reused across all rows
    HIDDEN: tl.constexpr = 768
    YBLOCK: tl.constexpr = 8
    XCOLS: tl.constexpr = 1024  # power-of-2 >= HIDDEN=768

    pid_y = tl.program_id(0)  # row-tile index
    row_start = pid_y * YBLOCK
    num_rows = xnumel // HIDDEN  # = 65536

    col_offs = tl.arange(0, XCOLS)   # [1024]
    col_mask = col_offs < HIDDEN       # True for cols 0..767

    # Load bias vectors ONCE per CTA - reused across all YBLOCK rows
    bias1 = tl.load(in_ptr1 + col_offs, mask=col_mask, other=0.0)
    bias2 = tl.load(in_ptr2 + col_offs, mask=col_mask, other=0.0)
    bias_sum = bias1 + bias2  # stays in registers

    # Process YBLOCK rows - statically unrolled loop
    for i in tl.static_range(YBLOCK):
        row = row_start + i
        row_valid = row < num_rows
        flat_offs = row * HIDDEN + col_offs  # [1024]
        mask2d = col_mask & row_valid
        v0 = tl.load(in_ptr0     + flat_offs, mask=mask2d, other=0.0)
        v2 = tl.load(in_out_ptr0 + flat_offs, mask=mask2d, other=0.0)
        v6 = tl.load(in_ptr3     + flat_offs, mask=mask2d, other=0.0)
        result = v0 + v2 + v6 + bias_sum
        tl.store(in_out_ptr0 + flat_offs, result, mask=mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_addmm_view_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 50331648, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
