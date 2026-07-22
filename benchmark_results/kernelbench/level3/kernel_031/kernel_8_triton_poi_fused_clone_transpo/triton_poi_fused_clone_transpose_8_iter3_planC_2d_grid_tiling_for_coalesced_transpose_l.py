import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # grid[0] = x1 head index (0..7); grid[1] = tiles over xnumel=524288 (inner elements)
    x1 = tl.program_id(0).to(tl.int64)           # head index, scalar per block
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]   # index into (x2*32 + x0) space
    xmask = xindex < xnumel                       # xnumel = 524288
    x0 = xindex % 32                              # feature dim [0..31]
    x2 = xindex // 32                             # seq dim [0..16383]
    # Load: all threads share same x1 -> 524288*x1 is a scalar offset
    # Within block: x0 varies by 1 (contiguous) -> fully coalesced
    src_offset = x0 + 32 * x2 + 524288 * x1
    tmp0 = tl.load(in_ptr0 + src_offset, mask=xmask, other=0.0)
    # Store: output layout (16384, 8, 32) strides (256, 32, 1)
    # out[x2, x1, x0] = x2*256 + x1*32 + x0
    dst_offset = x0 + 32 * x1 + 256 * x2
    tl.store(out_ptr0 + dst_offset, tmp0, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 8, 32), (256, 32, 1), device='cuda:0', dtype=torch.float32)
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
