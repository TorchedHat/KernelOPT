import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid_row = tl.program_id(0)   # 0..16383, covers x1
    pid_col = tl.program_id(1)   # 0..1, covers col//128
    lane = tl.arange(0, XBLOCK)  # 0..127

    # COALESCED load: 128 consecutive floats from same in_ptr0 row
    # in_ptr0 shape (32768, 384), strides (384, 1)
    # load_row = 2*pid_row + pid_col maps to row in [0..32767]
    load_row = 2 * pid_row + pid_col
    in_val = tl.load(in_ptr0 + load_row * 384 + lane)

    # Bias load: in_ptr1 shape (384,), bias index = col%128 = lane (0..127)
    bias_val = tl.load(in_ptr1 + lane, eviction_policy='evict_last')

    # Fused add + scale
    result = (in_val + bias_val) * 0.1767766952966369

    # Scatter store to out_ptr0 shape (8, 16384, 32), strides (524288, 32, 1)
    # col = pid_col*128 + lane ranges 0..127 (pid_col=0) or 128..255 (pid_col=1)
    col = pid_col * 128 + lane   # 0..255
    x2 = col // 32               # batch index 0..7
    x0 = col % 32                # innermost dim 0..31
    out_idx = x2 * 524288 + pid_row * 32 + x0
    tl.store(out_ptr0 + out_idx, result)

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
