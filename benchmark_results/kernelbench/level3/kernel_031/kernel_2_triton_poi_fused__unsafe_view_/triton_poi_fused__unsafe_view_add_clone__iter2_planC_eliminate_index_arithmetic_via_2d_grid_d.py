import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    pid_x = tl.program_id(0)   # x1: sequence position 0..16383
    pid_y = tl.program_id(1)   # combined col 0..255 = x2*32+x0

    # Decompose pid_y into x2 and x0 (cheap, done once per CTA)
    x2 = pid_y // 32            # 0..7
    x0 = pid_y % 32             # 0..31
    x1 = pid_x                  # 0..16383

    # col = x0 + 32*x2, range [0, 255]
    col = x0 + 32 * x2

    # col//256 == 0 always (max col = 255 < 256), so that term vanishes
    col_div_128 = col // 128    # 0 or 1
    col_mod_128 = col % 128     # 0..127

    # in_ptr0 shape: (32768, 384), strides: (384, 1)
    # Original: 384*(col//128) + 768*x1 + 768*(col//256) + (col%128)
    # Since col//256 == 0: = 384*col_div_128 + 768*x1 + col_mod_128
    in0_offset = 384 * col_div_128 + 768 * x1 + col_mod_128
    tmp0 = tl.load(in_ptr0 + in0_offset)

    # in_ptr1 shape: (384,), stride: (1,)
    tmp1 = tl.load(in_ptr1 + col_mod_128)

    SCALE: tl.constexpr = 0.1767766952966369
    tmp4 = tl.math.fma(tmp0, SCALE, tmp1 * SCALE)

    # out_ptr0 shape: (8, 16384, 32), strides: (524288, 32, 1)
    # flat index = x2*524288 + x1*32 + x0
    out_offset = x2 * 524288 + x1 * 32 + x0
    tl.store(out_ptr0 + out_offset, tmp4)

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
