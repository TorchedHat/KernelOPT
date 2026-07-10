import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK covers channel dim (0..127, stride=1 in in_ptr0) => COALESCED loads
    # YBLOCK covers batch*spatial dim (0..1007999)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape (1, XBLOCK): channel indices
    xmask = xindex < 128                                 # channel dim is exactly 128

    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape (YBLOCK, 1): batch*spatial indices
    ymask = yindex < 32 * 31500                          # 32 batches * 31500 spatial = 1008000

    # Decompose yindex into batch (y1) and spatial (x2)
    y1 = yindex // 31500    # batch index (0..31)
    x2 = yindex % 31500     # spatial index (0..31499), where spatial = h*250 + w

    # Channel index from xindex
    y0 = xindex             # channel index (0..127), shape (1, XBLOCK)

    # Load index: in_ptr0 has strides (4032000, 1, 32000, 128) for (batch, channel, H, W)
    # channel stride=1 => y0 varies fastest => COALESCED across warp threads
    # h = x2 // 250, w = x2 % 250
    load_idx = y0 + 4032000 * y1 + 32000 * (x2 // 250) + 128 * (x2 % 250)
    tmp0 = tl.load(in_ptr0 + load_idx, xmask & ymask)

    # Store index: out_ptr0 has strides (4032000, 31500, 250, 1) for (batch, channel, H, W)
    # x2 = h*250 + w => x2*1 gives correct offset for (H,W) with strides (250,1)
    store_idx = x2 + 31500 * y0 + 4032000 * y1
    tl.store(out_ptr0 + store_idx, tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
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
