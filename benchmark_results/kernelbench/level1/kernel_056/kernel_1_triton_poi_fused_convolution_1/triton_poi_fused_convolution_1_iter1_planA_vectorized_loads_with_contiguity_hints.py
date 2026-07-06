import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 4096   # channel*batch (stride-1 in src)
    ynumel = 31500  # spatial positions
    xoffset = tl.program_id(0) * XBLOCK
    # Apply contiguity hints on the 1D arange BEFORE broadcasting to 2D
    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xoffset + xarange[None, :]  # shape [1, XBLOCK] — fast dim = stride-1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < ynumel
    x0 = xindex % 128    # channel index (stride-1 in src)
    x1 = xindex // 128   # batch index
    y2 = yindex          # spatial index
    tmp0 = tl.load(in_ptr0 + (x0 + 128*y2 + 4032000*x1), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y2 + 31500*xindex), tmp0, ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
