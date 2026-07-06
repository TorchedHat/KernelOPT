import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK=128 covers all C channels (stride-1 in src) -> coalesced loads
    # XBLOCK=32  covers spatial x2 positions (stride-1 in dst) -> coalesced stores
    pid_x = tl.program_id(0)   # spatial tile index
    pid_y = tl.program_id(1)   # batch index (0..N-1)
    x2_offs = pid_x * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK], spatial indices
    x2_mask = x2_offs < xnumel                          # guard partial last tile (31500 % 32 = 28)
    y0_offs = tl.arange(0, YBLOCK)                      # [YBLOCK=128], all channels, no mask needed
    y1 = pid_y                                           # batch index (0..31)
    # Load: y0 is stride-1 in src -> consecutive y0 threads load consecutive addrs (coalesced)
    # load_addr shape: [YBLOCK, XBLOCK]
    load_addr = y0_offs[:, None] + 128 * x2_offs[None, :] + 4032000 * y1
    tile = tl.load(in_ptr0 + load_addr, mask=x2_mask[None, :])
    # Store: x2 is stride-1 in dst -> consecutive x2 threads store consecutive addrs (coalesced)
    # store_addr shape: [YBLOCK, XBLOCK]
    store_addr = x2_offs[None, :] + 31500 * y0_offs[:, None] + 4032000 * y1
    tl.store(out_ptr0 + store_addr, tile, mask=x2_mask[None, :])

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
