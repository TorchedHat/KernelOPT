import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D grid: total_numel = 32514048 passed as ynumel, xnumel=1, YBLOCK=XBLOCK, XBLOCK=1
    pid = tl.program_id(0)
    idx = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = idx < ynumel
    ow = idx % 63
    oh = (idx // 63) % 63
    c  = (idx // 3969) % 64
    n  = idx // 254016
    base = n * 4129024 + c + oh * 65024 + ow * 256
    v00 = tl.load(in_ptr0 + base,         mask=mask)
    v01 = tl.load(in_ptr0 + base + 64,    mask=mask)
    v02 = tl.load(in_ptr0 + base + 128,   mask=mask)
    v03 = tl.load(in_ptr0 + base + 192,   mask=mask)
    v10 = tl.load(in_ptr0 + base + 16256, mask=mask)
    v11 = tl.load(in_ptr0 + base + 16320, mask=mask)
    v12 = tl.load(in_ptr0 + base + 16384, mask=mask)
    v13 = tl.load(in_ptr0 + base + 16448, mask=mask)
    v20 = tl.load(in_ptr0 + base + 32512, mask=mask)
    v21 = tl.load(in_ptr0 + base + 32576, mask=mask)
    v22 = tl.load(in_ptr0 + base + 32640, mask=mask)
    v23 = tl.load(in_ptr0 + base + 32704, mask=mask)
    v30 = tl.load(in_ptr0 + base + 48768, mask=mask)
    v31 = tl.load(in_ptr0 + base + 48832, mask=mask)
    v32 = tl.load(in_ptr0 + base + 48896, mask=mask)
    v33 = tl.load(in_ptr0 + base + 48960, mask=mask)
    result = tl.maximum(tl.maximum(tl.maximum(tl.maximum(
               tl.maximum(tl.maximum(tl.maximum(tl.maximum(
               tl.maximum(tl.maximum(tl.maximum(tl.maximum(
               tl.maximum(tl.maximum(tl.maximum(v00, v01), v02), v03),
               v10), v11), v12), v13), v20), v21), v22), v23), v30), v31), v32), v33)
    out_idx = n * 254016 + c * 3969 + oh * 63 + ow
    tl.store(out_ptr0 + out_idx, result, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 508032
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3[grid](arg_0, arg_1, 508032, 64, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 63, 63), (254016, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
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
