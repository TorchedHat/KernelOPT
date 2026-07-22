import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16}, num_warps=1),
        triton.Config({'XBLOCK': 32}, num_warps=1),
        triton.Config({'XBLOCK': 64}, num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid_x = tl.program_id(0)   # column tile
    pid_y = tl.program_id(1)   # row index (0..9)
    col_numel = 128
    col_offset = pid_x * XBLOCK
    col_idx = col_offset + tl.arange(0, XBLOCK)[:]
    col_mask = col_idx < col_numel
    tmp0 = tl.load(in_ptr0 + col_idx, col_mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (col_idx + 256 * pid_y), tmp0, col_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_expand_unsqueeze_4[grid](arg_0, arg_1, 1280)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
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
