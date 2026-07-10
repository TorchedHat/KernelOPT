import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D grid: pid maps over all N*C*S = 129,024,000 elements
    # XBLOCK is used as BLOCK_SIZE (128), YBLOCK is unused (set to 1)
    # Flat index order: (b, s, c) so consecutive threads differ in c (stride=1 in input)
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = 129024000  # 32 * 128 * 31500
    mask = offsets < numel
    # Decompose: innermost=c (0..127), middle=s (0..31499), outer=b (0..31)
    c = offsets % 128
    tmp = offsets // 128
    s = tmp % 31500
    b = tmp // 31500
    # Reconstruct h, w from flat spatial index s
    h = s // 250
    w = s % 250
    # Load index: b*4032000 + c*1 + h*32000 + w*128
    # Consecutive threads differ by c -> stride=1 -> COALESCED
    load_idx = b * 4032000 + c + h * 32000 + w * 128
    tmp0 = tl.load(in_ptr0 + load_idx, mask=mask, eviction_policy='evict_last')
    # Store index: b*4032000 + c*31500 + s (NCHW contiguous output)
    store_idx = b * 4032000 + c * 31500 + s
    tl.store(out_ptr0 + store_idx, tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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
