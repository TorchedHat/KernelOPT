import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK covers ynumel=512 (channel*batch) — output stride=1 in channel dim
    # YBLOCK covers xnumel=8192 (spatial H*W)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # covers ynumel=512
    xmask = xindex < 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # covers xnumel=8192
    # Decompose xindex: channel (fast, output stride=1) and batch (slow)
    x_ch  = (xindex % 64)   # channel in [0,64), output stride=1
    x_bat = xindex // 64    # batch   in [0,8)
    # Decompose yindex: spatial W (fast) and spatial H (slow)
    y_w   = (yindex % 256)  # spatial W in [0,256)
    y_h   = yindex // 256   # spatial H in [0,32)
    y_sp  = yindex          # flat spatial index for output
    # Input loads — same math as before with variable renaming
    base = 2*y_w + 1024*y_h + 32768*x_ch + 4194304*x_bat
    tmp0 = tl.load(in_ptr0 + base,       mask=xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + base + 1,   mask=xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + base + 512, mask=xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + base + 513, mask=xmask, eviction_policy='evict_first')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    # Output store: adjacent threads differ by 1 in x_ch -> stride-1 -> COALESCED
    tl.store(out_ptr0 + (x_ch + 64*y_sp + 524288*x_bat), tmp6, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 512
    xnumel = 8192
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5[grid](arg_0, arg_1, 512, 8192, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 64, 32, 256), (524288, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
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
