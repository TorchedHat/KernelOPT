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
    # 2D grid: program_id(0) = column tile, program_id(1) = row tile
    # x0 = column index in [0, 127]; x1 = row index in [0, 9]
    xoffset = tl.program_id(0) * XBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    x1 = tl.program_id(1) + tl.zeros([1, 1], dtype=tl.int32)  # shape [1, 1], scalar row
    xmask = x0 < xnumel  # guard columns < 128
    # Broadcast load: in_ptr0 has strides (128,128,1); load same 128 values for every row
    tmp0 = tl.load(in_ptr0 + x0, mask=xmask, eviction_policy='evict_last')  # [1, XBLOCK]
    # Strided store: out_ptr0 has strides (256,128,1); row stride = 256
    tl.store(out_ptr0 + (x0 + 256 * x1), tmp0, mask=xmask)

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
