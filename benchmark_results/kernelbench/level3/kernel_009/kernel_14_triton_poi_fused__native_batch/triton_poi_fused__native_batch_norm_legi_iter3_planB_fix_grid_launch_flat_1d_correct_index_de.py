import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D kernel: treat YBLOCK*XBLOCK as BLOCK_SIZE, use only pid(0)
    # numel = 589824, BLOCK_SIZE = YBLOCK*XBLOCK = 8*128 = 1024
    BLOCK_SIZE: tl.constexpr = YBLOCK * XBLOCK
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = ynumel * xnumel  # 65536 * 9 = 589824
    mask = i < numel
    # Decompose flat index into (x2, y3, y0, y1)
    x2 = i % 9          # spatial kernel position 0..8
    y3 = i // 9         # combined channel index 0..65535
    y0 = y3 % 256       # channel-in 0..255
    y1 = y3 // 256      # channel-out 0..255
    # Load: x2 + 9*y3 = (i%9) + 9*(i//9) = i  =>  load in_ptr0[i] directly
    tmp0 = tl.load(in_ptr0 + i, mask=mask)
    # Store: matches original formula exactly
    tl.store(out_ptr0 + (y0 + 256 * x2 + 2304 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 65536
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 65536, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
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
