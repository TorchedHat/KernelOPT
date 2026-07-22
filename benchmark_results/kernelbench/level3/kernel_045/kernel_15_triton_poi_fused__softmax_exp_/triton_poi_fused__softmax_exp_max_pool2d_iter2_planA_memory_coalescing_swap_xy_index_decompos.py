import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # covers ynumel=2048 (channel*batch)
    xmask = xindex < 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # covers xnumel=512 (spatial)
    ymask = yindex < 512
    xmask = xmask & ymask

    y0 = (xindex % 256)  # channel (0..255) — fast dim, stride-1 in output
    y1 = xindex // 256   # batch   (0..7)
    x2 = (yindex % 64)   # W_out half-index (0..63)
    x3 = yindex // 64    # H_out half-index (0..7)

    tmp0 = tl.load(in_ptr0 + (2*x2 + 256*x3 + 2048*y0 + 1048576*y1),       mask=xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1),   mask=xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), mask=xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), mask=xmask, eviction_policy='evict_first')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 256*yindex + 131072*y1), tmp6, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 2048
    xnumel = 512
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15[grid](arg_0, arg_1, 2048, 512, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 256, 8, 64), (131072, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
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
