import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent kernel: launch with grid=(132,) i.e. one CTA per SM
    # XBLOCK is repurposed as BLOCK_SIZE (elements per iteration)
    # NUM_SMS is hardcoded to 132 (H200 SXM5)
    NUM_SMS: tl.constexpr = 132
    total_numel: tl.constexpr = 179712  # 19968 * 9
    xnumel_val: tl.constexpr = 9

    pid = tl.program_id(0)
    tile_id = pid
    while tile_id * XBLOCK < total_numel:
        flat_start = tile_id * XBLOCK
        offsets = flat_start + tl.arange(0, XBLOCK)
        mask = offsets < total_numel
        # Decompose flat index into 2D (y3, x2)
        y3 = offsets // xnumel_val   # row index in [0, 19968)
        x2 = offsets % xnumel_val    # col index in [0, 9)
        # Further decompose y3 into channel (y0) and batch (y1)
        y0 = y3 % 96
        y1 = y3 // 96
        # Load: input is row-major, stride=(864,9,3,1) -> x2 + 9*y3
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask)
        # Store: output is transposed, stride=(864,1,288,96) -> y0 + 96*x2 + 864*y1
        tl.store(out_ptr0 + (y0 + 96 * x2 + 864 * y1), tmp0, mask=mask)
        tile_id += NUM_SMS   # each SM strides to its next tile

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 19968
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_34[grid](arg_0, arg_1, 19968, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((208, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((208, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
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
