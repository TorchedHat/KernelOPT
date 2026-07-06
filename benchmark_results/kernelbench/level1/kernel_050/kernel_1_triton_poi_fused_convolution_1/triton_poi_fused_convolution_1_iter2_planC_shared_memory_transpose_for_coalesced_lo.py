import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Input:  [N=32, C=128, H=126, W=250], strides [4032000, 1, 32000, 128]
    # Output: [N=32, C=128, H=126, W=250], strides [4032000, 31500, 250, 1]
    # xnumel = 31500 (spatial positions H*W = 126*250)
    # ynumel = 4096  (N*C = 32*128)
    # YBLOCK repurposed as TILE_C (=128, all channels per CTA)
    # XBLOCK repurposed as TILE_X (spatial tile size)
    pid_y = tl.program_id(1)   # batch index n (0..31)
    pid_x = tl.program_id(0)   # spatial tile index

    # Spatial indices: stride=1 in output => coalesced store dim
    x_base = pid_x * XBLOCK
    x_idx = x_base + tl.arange(0, XBLOCK)   # shape [XBLOCK]
    x_mask = x_idx < xnumel

    # Channel indices 0..127: stride=1 in input => coalesced load dim
    c_idx = tl.arange(0, YBLOCK)             # shape [YBLOCK=128]

    # Phase 1: COALESCED LOAD
    # Load tile shape: [XBLOCK, YBLOCK] — c_idx is the LAST (innermost) dim
    # In input: element offset = n*4032000 + c*1 + x_idx*128
    # c_idx innermost (stride=1 in input) => coalesced loads
    load_offsets = x_idx[:, None] * 128 + c_idx[None, :] + 4032000 * pid_y
    # load_offsets shape: [XBLOCK, YBLOCK]
    load_mask = x_mask[:, None]  # broadcast over YBLOCK, shape [XBLOCK, YBLOCK]
    tile = tl.load(in_ptr0 + load_offsets, mask=load_mask, other=0.0)
    # tile shape: [XBLOCK, YBLOCK]

    # Phase 2: COALESCED STORE
    # Store tile shape: [YBLOCK, XBLOCK] — x_idx is the LAST (innermost) dim
    # In output: element offset = n*4032000 + c*31500 + x_idx
    # x_idx innermost (stride=1 in output) => coalesced stores
    # Transpose tile from [XBLOCK, YBLOCK] to [YBLOCK, XBLOCK]
    tile_t = tl.trans(tile)
    # tile_t shape: [YBLOCK, XBLOCK]
    store_offsets = c_idx[:, None] * 31500 + x_idx[None, :] + 4032000 * pid_y
    # store_offsets shape: [YBLOCK, XBLOCK]
    store_mask = x_mask[None, :]  # broadcast over YBLOCK, shape [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_offsets, tile_t, mask=store_mask)

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
