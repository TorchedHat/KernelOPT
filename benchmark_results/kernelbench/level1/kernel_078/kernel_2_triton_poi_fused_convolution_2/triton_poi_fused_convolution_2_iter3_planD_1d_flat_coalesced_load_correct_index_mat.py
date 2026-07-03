import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # 1D flat kernel: numel = ynumel * xnumel = 24576 * 3025 = 74317824
    # YBLOCK is always 1 in configs; XBLOCK is the actual tile size (power-of-2)
    # Flat index f ordered so y0 (channel, stride-1 in in_ptr0) is fast dim
    pid = tl.program_id(0)
    f = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = ynumel * xnumel  # 74317824
    mask = f < numel
    # Decompose: y0=channel(stride-1 in in_ptr0), x2=spatial, y1=batch
    y0 = f % 96                 # FAST DIM: stride-1 in in_ptr0 -> COALESCED
    tmp_f = f // 96
    x2 = tmp_f % 3025           # spatial index
    y1 = tmp_f // 3025          # batch index
    y3 = y0 + 96 * y1           # = original yindex
    # COALESCED: adjacent lanes differ by 1 in y0 (stride-1)
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), mask, eviction_policy='evict_first')
    # COALESCED: adjacent lanes access adjacent bias elements
    tmp1 = tl.load(in_ptr1 + y0, mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store: same index as original
    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
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
