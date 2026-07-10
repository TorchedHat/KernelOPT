import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Input  shape (32,128,126,250), strides (4032000,1,32000,128)
    # Output shape (32,128,126,250), strides (4032000,31500,250,1)
    # ynumel=4096=32*128, xnumel=31500=126*250
    #
    # Repurpose: XBLOCK=TILE_C (channel, stride-1 in input), YBLOCK=TILE_S (spatial, stride-1 in output)
    # 2D grid: axis 0 = channel tiles (pid_c), axis 1 = batch*spatial tiles (pid_bs)
    #
    pid_c = tl.program_id(0)   # channel tile
    pid_bs = tl.program_id(1)  # batch * spatial tile (combined)

    # Decompose pid_bs into batch index and spatial tile
    num_s_tiles = tl.cdiv(31500, YBLOCK)
    b_idx = pid_bs // num_s_tiles   # batch index (0..31)
    pid_s = pid_bs % num_s_tiles    # spatial tile index

    # Channel offsets - innermost for LOAD (stride=1 in input)
    c_off = pid_c * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK=TILE_C]
    c_mask = c_off < 128

    # Spatial offsets - innermost for STORE (stride=1 in output)
    s_off = pid_s * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK=TILE_S]
    s_mask = s_off < 31500

    # COALESCED LOAD: consecutive threads vary c_off (stride=1 in input)
    # flat address = channel + 128*spatial + 4032000*batch
    in_addr = c_off[None, :] + 128 * s_off[:, None] + 4032000 * b_idx
    full_mask = s_mask[:, None] & c_mask[None, :]
    tile = tl.load(in_ptr0 + in_addr, mask=full_mask, other=0.0)
    # tile shape: [YBLOCK, XBLOCK]

    # COALESCED STORE: consecutive threads vary s_off (stride=1 in output)
    # flat address = spatial + 31500*channel + 4032000*batch
    out_addr = s_off[:, None] + 31500 * (c_off[None, :] + 128 * b_idx)
    tl.store(out_ptr0 + out_addr, tile, mask=full_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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
