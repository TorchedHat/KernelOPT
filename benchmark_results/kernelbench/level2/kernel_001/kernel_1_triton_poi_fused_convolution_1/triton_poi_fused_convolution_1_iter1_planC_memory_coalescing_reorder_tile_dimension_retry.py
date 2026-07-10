import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # XBLOCK covers channel dim (0..127, stride=1 in input) -- innermost/lane dim -> COALESCED
    # YBLOCK covers spatial dim (0..31499, stride=128 in input) -- outer dim
    # grid dim 2 covers batch (0..31)

    # Channel dimension: innermost warp dimension (stride=1 in input)
    xoffset = tl.program_id(0) * XBLOCK
    channel = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    xmask = channel < 128  # shape [1, XBLOCK]

    # Spatial dimension: outer tile dimension (stride=128 in input)
    yoffset = tl.program_id(1) * YBLOCK
    spatial = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = spatial < 31500  # shape [YBLOCK, 1]

    # Batch dimension: one block per batch element
    batch = tl.program_id(2)  # scalar, 0..31

    # Combined mask broadcasts to [YBLOCK, XBLOCK]
    mask = xmask & ymask

    # Load: in[batch, channel, h, w] with stride (4032000, 1, 32000, 128)
    # spatial = h*250 + w  =>  128*spatial = h*32000 + w*128
    # offset = batch*4032000 + channel*1 + h*32000 + w*128
    #        = batch*4032000 + channel + 128*spatial
    tmp0 = tl.load(in_ptr0 + (channel + 128 * spatial + 4032000 * batch),
                   mask, eviction_policy='evict_last')

    # Store: out[batch, channel, h, w] with stride (4032000, 31500, 250, 1)
    # offset = batch*4032000 + channel*31500 + h*250 + w
    #        = batch*4032000 + channel*31500 + spatial
    tl.store(out_ptr0 + (spatial + 31500 * channel + 4032000 * batch),
             tmp0, mask)

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
