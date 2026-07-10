import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK: tl.constexpr):
    SPATIAL_SIZE: tl.constexpr = 53816
    NUM_SPATIAL_TILES: tl.constexpr = 421   # cdiv(53816, 128)
    TOTAL_TILES: tl.constexpr = 861056      # 421 * 2048
    cta_id = tl.program_id(0)
    total_ctas = tl.num_programs(0)
    tile_id = cta_id
    while tile_id < TOTAL_TILES:
        pid_c = tile_id // NUM_SPATIAL_TILES  # batch*channel index (0..2047)
        pid_s = tile_id % NUM_SPATIAL_TILES   # spatial tile index
        # Channel scalar loads - channels cycle with period 16
        chan_idx = pid_c % 16
        tmp1 = tl.load(in_ptr0 + chan_idx)
        tmp3 = tl.load(in_ptr1 + chan_idx)
        tmp6 = tl.load(in_ptr2 + chan_idx)
        # Spatial tile with boundary mask (53816 not divisible by 128)
        sidx = pid_s * XBLOCK + tl.arange(0, XBLOCK)[:]
        xmask = sidx < SPATIAL_SIZE
        flat_idx = pid_c * SPATIAL_SIZE + sidx
        tmp0 = tl.load(in_out_ptr0 + flat_idx, xmask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = libdevice.tanh(tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.sigmoid(tmp7)
        tl.store(in_out_ptr0 + flat_idx, tmp8, xmask)
        tile_id += total_ctas

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
