import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK = TILE_C (channel tile), XBLOCK = TILE_HW (spatial tile)
    # Input layout:  (N=32, C=128, H=126, W=250), strides=(4032000, 1, 32000, 128)
    #   => in[n,c,h,w] at n*4032000 + c*1 + h*32000 + w*128
    #   => with hw_off = h*250+w: in[n,c,hw_off] at n*4032000 + c + hw_off*128
    # Output layout: (N=32, C=128, H=126, W=250), strides=(4032000, 31500, 250, 1)
    #   => out[n,c,h,w] at n*4032000 + c*31500 + h*250 + w
    #   => with hw_off = h*250+w: out[n,c,hw_off] at n*4032000 + c*31500 + hw_off
    C  = 128
    HW = 31500
    N  = 32
    pid_hw = tl.program_id(0)   # tile over HW dimension
    pid_c  = tl.program_id(1)   # tile over C dimension
    pid_n  = tl.program_id(2)   # tile over N (batch) dimension

    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)   # shape [XBLOCK]
    c_off  = pid_c  * YBLOCK + tl.arange(0, YBLOCK)   # shape [YBLOCK]

    hw_mask = hw_off < HW   # shape [XBLOCK]
    c_mask  = c_off  < C    # shape [YBLOCK]

    # 2D tile: [XBLOCK, YBLOCK] = [TILE_HW, TILE_C]
    # mask shape: [XBLOCK, YBLOCK]
    mask = hw_mask[:, None] & c_mask[None, :]

    n_base = pid_n * 4032000

    # COALESCED LOAD: innermost dim is c_off (stride=1 in input)
    # load_ptrs[i, j] = in_ptr0 + n_base + hw_off[i]*128 + c_off[j]
    # Consecutive j (c_off) differ by 1 => coalesced
    load_ptrs = in_ptr0 + n_base + hw_off[:, None] * 128 + c_off[None, :]
    tile = tl.load(load_ptrs, mask=mask, other=0.0)

    # COALESCED STORE: innermost dim is hw_off (stride=1 in output)
    # store_ptrs[i, j] = out_ptr0 + n_base + c_off[j]*31500 + hw_off[i]
    # Consecutive i (hw_off) differ by 1 => coalesced
    store_ptrs = out_ptr0 + n_base + c_off[None, :] * 31500 + hw_off[:, None]
    tl.store(store_ptrs, tile, mask=mask)

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
