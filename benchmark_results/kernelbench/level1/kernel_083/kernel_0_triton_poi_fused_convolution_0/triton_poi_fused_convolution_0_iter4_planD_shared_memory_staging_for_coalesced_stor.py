import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # y-axis: batch index (0..9), x-axis: output flat index (0..150527)
    # out_flat = channel + 3*spatial  =>  spatial = out_flat // 3, channel = out_flat % 3
    # input address:  spatial + 50176 * (batch*3 + channel)  [NCHW layout]
    # output address: batch*150528 + out_flat  (stride-1 across warp lanes, coalesced)
    batch_numel = 10
    out_numel = 150528  # 3 * 50176
    yoffset = tl.program_id(1) * YBLOCK
    batch_idx = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    bmask = batch_idx < batch_numel
    xoffset = tl.program_id(0) * XBLOCK
    out_flat = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]  # [1, XBLOCK]
    omask = out_flat < out_numel
    channel = out_flat % 3          # compile-time C=3: compiler emits multiply-shift
    spatial = out_flat // 3         # compile-time C=3: compiler emits multiply-shift
    in_addr = spatial + 50176 * (batch_idx * 3 + channel)  # [YBLOCK, XBLOCK]
    val = tl.load(in_ptr0 + in_addr, bmask & omask)
    tl.store(out_ptr0 + (batch_idx * 150528 + out_flat), val, bmask & omask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
