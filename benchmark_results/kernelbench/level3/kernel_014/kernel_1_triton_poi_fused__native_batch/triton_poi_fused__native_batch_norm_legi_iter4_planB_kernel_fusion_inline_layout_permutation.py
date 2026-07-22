import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel over all 9216 elements (32*32*3*3)
    # YBLOCK is fixed to 1 in all configs; XBLOCK drives the tile size
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    idx = xoffset + tl.arange(0, XBLOCK)
    mask = idx < 9216
    # Decompose flat index: y3 = output channel pair index, x2 = spatial index
    y3 = idx // 9
    x2 = idx % 9
    y0 = y3 % 32
    y1 = y3 // 32
    # Load from in_ptr0 contiguously (in_ptr0 has strides (288,9,3,1), flat idx = x2 + 9*y3)
    tmp0 = tl.load(in_ptr0 + idx, mask=mask, eviction_policy='evict_first')
    # Store to out_ptr0 with permuted layout (strides 288,1,96,32)
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, mask=mask, eviction_policy='evict_last')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1024
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
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
