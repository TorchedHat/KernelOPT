import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Tensor layout:
    # in_ptr0:  (N=32, C=128, H=126, W=250), strides (4032000, 1, 32000, 128)  [channels-last]
    # out_ptr0: (N=32, C=128, H=126, W=250), strides (4032000, 31500, 250, 1)  [NCHW]
    #
    # Strategy: YBLOCK tiles over C (stride=1 in input -> coalesced load)
    #           XBLOCK tiles over W (stride=1 in output -> coalesced store)
    # Grid: (cdiv(W,XBLOCK)*cdiv(C,YBLOCK), H, N) mapped to pid(0), pid(1), pid(2)
    # But we use 2D grid: pid(0) = CW tile, pid(1) = H*N combined
    #
    # Use YBLOCK as CBLOCK (innermost for loads, stride=1 in input)
    # Use XBLOCK as WBLOCK (innermost for stores, stride=1 in output)
    # Constants: N=32, C=128, H=126, W=250

    N_val = 32
    C_val = 128
    H_val = 126
    W_val = 250

    # pid(0) covers CW tiles: num_w_tiles = cdiv(W, XBLOCK)
    # pid(1) covers H*N: h_idx = pid1 % H_val, n_idx = pid1 // H_val
    pid_cw = tl.program_id(0)
    pid_hn = tl.program_id(1)

    num_w_tiles = tl.cdiv(W_val, XBLOCK)
    pid_c = pid_cw // num_w_tiles
    pid_w = pid_cw %  num_w_tiles

    h_idx = pid_hn % H_val
    n_idx = pid_hn // H_val

    # YBLOCK tiles over C (stride=1 in input)
    c_off = pid_c * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK]
    # XBLOCK tiles over W (stride=1 in output)
    w_off = pid_w * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK]

    c_mask = c_off < C_val   # [YBLOCK]
    w_mask = w_off < W_val   # [XBLOCK]
    mask2d = c_mask[:, None] & w_mask[None, :]  # [YBLOCK, XBLOCK]

    # Input: strides (4032000, 1, 32000, 128)
    # in[n,c,h,w] = n*4032000 + c*1 + h*32000 + w*128
    # C is innermost (stride=1) -> coalesced along YBLOCK dim
    in_base = n_idx * 4032000 + h_idx * 32000
    in_ptrs = in_ptr0 + in_base + c_off[:, None] * 1 + w_off[None, :] * 128
    tmp = tl.load(in_ptrs, mask=mask2d)  # [YBLOCK, XBLOCK]

    # Output: strides (4032000, 31500, 250, 1)
    # out[n,c,h,w] = n*4032000 + c*31500 + h*250 + w*1
    # W is innermost (stride=1) -> coalesced along XBLOCK dim
    out_base = n_idx * 4032000 + h_idx * 250
    out_ptrs = out_ptr0 + out_base + c_off[:, None] * 31500 + w_off[None, :] * 1
    tl.store(out_ptrs, tmp, mask=mask2d)  # [YBLOCK, XBLOCK]

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
