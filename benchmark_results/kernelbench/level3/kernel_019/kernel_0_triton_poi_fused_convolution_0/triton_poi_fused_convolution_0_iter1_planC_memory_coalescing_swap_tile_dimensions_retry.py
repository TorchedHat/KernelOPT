import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4, 'YBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 2, 'YBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 2, 'YBLOCK': 128}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # X-tile covers channel (0..2), stride-1 in OUTPUT -> coalesced stores
    # Y-tile covers batch*spatial (0..501759)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # channel index: 0..2
    xmask = xindex < 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # batch*spatial index: 0..501759
    ymask = yindex < 501760
    channel = xindex            # 0..2, shape [1, XBLOCK]
    spatial = yindex % 50176    # 0..50175, shape [YBLOCK, 1]
    batch   = yindex // 50176   # 0..9, shape [YBLOCK, 1]
    # Input layout: [batch, channel, spatial] with strides [150528, 50176, 1]
    tmp0 = tl.load(in_ptr0 + (batch * 150528 + channel * 50176 + spatial), xmask & ymask, eviction_policy='evict_last')
    # Output layout: [batch, spatial, channel] with strides [150528, 3, 1] -> channel stride-1 -> COALESCED
    tl.store(out_ptr0 + (batch * 150528 + spatial * 3 + channel), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
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
