import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # 3D-grid coalesced transpose kernel
    # Grid: (cdiv(128,XBLOCK), cdiv(31500,YBLOCK), 32)
    # pid_c: channel tile (fast axis = channel, stride-1 in input -> coalesced load)
    # pid_x: spatial tile (fast axis = spatial, stride-1 in output -> coalesced store)
    # pid_n: batch index
    pid_c = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_n = tl.program_id(2)

    c_off = pid_c * XBLOCK
    x_off = pid_x * YBLOCK

    tc = tl.arange(0, XBLOCK)  # [XBLOCK] channel offsets within tile
    tx = tl.arange(0, YBLOCK)  # [YBLOCK] spatial offsets within tile

    c_idx = c_off + tc[:, None]   # [XBLOCK, 1] absolute channel indices
    x_idx = x_off + tx[None, :]   # [1, YBLOCK] absolute spatial indices

    c_mask = c_idx < 128
    x_mask = x_idx < 31500
    mask = c_mask & x_mask

    # Decompose flat spatial index x_idx into (h, w): H=126, W=250
    h_idx = x_idx // 250   # [1, YBLOCK]
    w_idx = x_idx % 250    # [1, YBLOCK]

    # INPUT strides: (N=4032000, C=1, H=32000, W=128)
    # load_addr = n*4032000 + c*1 + h*32000 + w*128
    # Fast axis = c (stride-1) -> consecutive tc-threads are COALESCED
    load_addr = (pid_n * 4032000
                 + c_idx * 1
                 + h_idx * 32000
                 + w_idx * 128)
    tile = tl.load(in_ptr0 + load_addr, mask=mask, other=0.0)

    # OUTPUT strides: (N=4032000, C=31500, spatial=1)
    # store_addr = n*4032000 + c*31500 + x2
    # Fast axis = x (stride-1) -> consecutive tx-threads are COALESCED
    store_addr = (pid_n * 4032000
                  + c_idx * 31500
                  + x_idx * 1)
    tl.store(out_ptr0 + store_addr, tile, mask=mask)

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
