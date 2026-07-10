import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK: tl.constexpr):
    CHANNEL_STRIDE: tl.constexpr = 53816
    NUM_CHANNELS: tl.constexpr = 16
    TILES_PER_CTA: tl.constexpr = 8
    pid = tl.program_id(0)
    tile_base = pid * TILES_PER_CTA
    for i in tl.static_range(TILES_PER_CTA):
        tile_idx = tile_base + i
        xoffset = tile_idx * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        # Per-element channel index (correct for all tiles including boundary-crossing ones)
        x1 = (xindex // CHANNEL_STRIDE) % NUM_CHANNELS
        # Load all 16 channel scalars as a vector, then gather by per-element channel index
        chan_mask = tl.full([NUM_CHANNELS], True, tl.int1)
        chan_bias   = tl.load(in_ptr0 + tl.arange(0, NUM_CHANNELS), mask=chan_mask)
        chan_scale  = tl.load(in_ptr1 + tl.arange(0, NUM_CHANNELS), mask=chan_mask)
        chan_scale2 = tl.load(in_ptr2 + tl.arange(0, NUM_CHANNELS), mask=chan_mask)
        tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
        tmp1 = tl.gather(chan_bias,   x1, 0)
        tmp3 = tl.gather(chan_scale,  x1, 0)
        tmp6 = tl.gather(chan_scale2, x1, 0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = libdevice.tanh(tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.sigmoid(tmp7)
        tl.store(in_out_ptr0 + xindex, tmp8, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 110215168
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_mul_sigmoid_tanh_0[grid](arg_0, arg_1, arg_2, arg_3, 110215168, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (861056, 53816, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
