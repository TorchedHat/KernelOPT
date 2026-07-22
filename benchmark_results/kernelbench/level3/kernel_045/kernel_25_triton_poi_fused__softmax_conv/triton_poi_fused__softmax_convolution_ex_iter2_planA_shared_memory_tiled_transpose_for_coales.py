import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_25(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Tiled transpose kernel: 3D grid (cdiv(512,XBLOCK), cdiv(512,YBLOCK), 8)
    # pid_x -> x2 tile (spatial 0..511), pid_y -> y0 tile (channel 0..511), pid_z -> y1 (batch 0..7)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    tx = tl.arange(0, XBLOCK)   # [XBLOCK] covers x2 within tile
    ty = tl.arange(0, YBLOCK)   # [YBLOCK] covers y0 within tile

    x2_base = pid_x * XBLOCK
    y0_base = pid_y * YBLOCK
    y1 = pid_z

    x2 = x2_base + tx   # [XBLOCK] global x2 indices
    y0 = y0_base + ty   # [YBLOCK] global y0 indices

    x2_mask = x2 < 512
    y0_mask = y0 < 512

    # COALESCED LOAD: in_ptr0 addr = y0 + 512*x2 + 262144*y1
    # Load as [XBLOCK, YBLOCK]: outer=tx(x2), inner=ty(y0)
    # Consecutive threads step by 1 in y0 (stride-1) -> coalesced!
    load_idx = tx[:, None] * 512 + ty[None, :] + x2_base * 512 + y0_base + 262144 * y1
    load_mask = x2_mask[:, None] & y0_mask[None, :]
    tile_data = tl.load(in_ptr0 + load_idx, load_mask, other=0.0)
    # tile_data shape: [XBLOCK, YBLOCK]

    # IN-REGISTER TRANSPOSE: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    tile_t = tl.trans(tile_data)
    # tile_t shape: [YBLOCK, XBLOCK]

    # Add bias: in_ptr1[y0], broadcast over x2 dimension
    bias = tl.load(in_ptr1 + y0, y0_mask, other=0.0)  # [YBLOCK]
    tile_t = tile_t + bias[:, None]

    # COALESCED STORE: out_ptr0 addr = x2 + 512*y0 + 524288*y1
    # Store as [YBLOCK, XBLOCK]: outer=ty(y0), inner=tx(x2)
    # Consecutive threads step by 1 in x2 (stride-1) -> coalesced!
    store_idx = tx[None, :] + x2_base + 512 * (ty[:, None] + y0_base) + 524288 * y1
    store_mask = x2_mask[None, :] & y0_mask[:, None]
    tl.store(out_ptr0 + store_idx, tile_t, store_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 4096
    xnumel = 512
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_25[grid](arg_0, arg_1, arg_2, 4096, 512, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 512, 8, 64), (262144, 1, 32768, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 512, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
