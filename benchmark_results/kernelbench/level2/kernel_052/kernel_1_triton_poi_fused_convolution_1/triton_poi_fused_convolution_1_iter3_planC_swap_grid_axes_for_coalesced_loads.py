import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1008000  # batch * spatial = 32 * 31500
    xnumel = 128      # channels
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    chan = xindex                          # channel index 0..127, stride-1 in input
    spatial_batch = yindex                 # combined index 0..1007999
    batch = spatial_batch // 31500         # batch index 0..31
    spatial = spatial_batch % 31500        # spatial index 0..31499
    combined_mask = xmask & ymask
    # Load: in_ptr0 strides (4032000, 1, 32000, 128)
    # in_ptr0[n,c,h,w] = n*4032000 + c*1 + h*32000 + w*128
    # spatial = h*250 + w => h=spatial//250, w=spatial%250
    # => batch*4032000 + chan + (spatial//250)*32000 + (spatial%250)*128
    # => batch*4032000 + chan + 128*spatial  (since 32000=128*250)
    tmp0 = tl.load(in_ptr0 + (chan + 128*spatial + 4032000*batch), mask=combined_mask, eviction_policy='evict_first')
    # Store: out_ptr0 strides (4032000, 31500, 250, 1)
    # out_ptr0[n,c,h,w] = n*4032000 + c*31500 + h*250 + w
    # => batch*4032000 + chan*31500 + spatial  (since 31500=250*126, spatial=h*250+w)
    tl.store(out_ptr0 + (spatial + 31500*chan + 4032000*batch), tmp0, mask=combined_mask)

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
