import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D indexing over all ynumel*xnumel = 134217728 output elements
    # YBLOCK*XBLOCK = BLOCK_SIZE (total elements per block)
    pid = tl.program_id(0)
    offsets = pid * (YBLOCK * XBLOCK) + tl.arange(0, YBLOCK * XBLOCK)
    numel = ynumel * xnumel
    mask = offsets < numel
    # Reconstruct (yindex, xindex) from flat offset
    # flat = yindex * 64 + xindex  =>  yindex = flat // 64, xindex = flat % 64
    x3 = offsets % 64          # xindex in [0, 64)
    yindex = offsets // 64     # yindex in [0, 2097152)
    y0 = yindex % 128
    y4 = yindex // 128
    y2 = yindex // 16384
    y5 = yindex % 16384
    # Load 4 pooling inputs (same addresses as original)
    tmp0 = tl.load(in_ptr0 + (x3 + 128*y0 + 32768*y4), mask)
    tmp1 = tl.load(in_ptr0 + (64 + x3 + 128*y0 + 32768*y4), mask)
    tmp3 = tl.load(in_ptr0 + (16384 + x3 + 128*y0 + 32768*y4), mask)
    tmp5 = tl.load(in_ptr0 + (16448 + x3 + 128*y0 + 32768*y4), mask)
    tmp8 = (tmp0 + tmp1 + tmp3 + tmp5) * 0.25  # divide by 4
    # Store using same formula as original
    tl.store(out_ptr0 + (y5 + 16384*x3 + 1048576*y2), tmp8, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 2097152
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2[grid](arg_0, arg_1, 2097152, 64, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8388608, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
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
