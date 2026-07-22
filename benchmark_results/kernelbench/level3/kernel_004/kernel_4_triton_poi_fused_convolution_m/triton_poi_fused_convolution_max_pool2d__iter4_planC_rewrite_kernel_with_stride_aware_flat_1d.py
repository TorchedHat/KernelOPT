import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Output: shape (4096,16,5,5), strides (400,25,5,1)
    # Input:  shape (4096,16,10,10), strides (1600,1,160,16)
    # 2x2 max-pool stride 2 — flat-1D indexing, numel passed as ynumel, BLOCK_SIZE as YBLOCK
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < ynumel
    # Decompose flat output index -> (n, c, oh, ow)
    ow = offsets % 5
    tmp = offsets // 5
    oh = tmp % 5
    tmp2 = tmp // 5
    c = tmp2 % 16
    n = tmp2 // 16
    # Input base: n*1600 + c*1 + (oh*2)*160 + (ow*2)*16 = n*1600 + c + oh*320 + ow*32
    in_base = n * 1600 + c + oh * 320 + ow * 32
    # Load 2x2 window (offsets: +0, +16, +160, +176)
    v00 = tl.load(in_ptr0 + in_base,       mask=mask, eviction_policy='evict_last')
    v01 = tl.load(in_ptr0 + in_base + 16,  mask=mask, eviction_policy='evict_last')
    v10 = tl.load(in_ptr0 + in_base + 160, mask=mask, eviction_policy='evict_last')
    v11 = tl.load(in_ptr0 + in_base + 176, mask=mask, eviction_policy='evict_last')
    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    # Output address: n*400 + c*25 + oh*5 + ow
    out_addr = n * 400 + c * 25 + oh * 5 + ow
    tl.store(out_ptr0 + out_addr, result, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 102400
    xnumel = 16
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 102400, 16, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 16, 10, 10), (1600, 1, 160, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
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
