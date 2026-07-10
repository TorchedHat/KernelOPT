import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=8, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    pid_x = tl.program_id(0)  # tiles over x2 (xnumel=31500)
    pid_y = tl.program_id(1)  # tiles over y3 (ynumel=4096)
    tx = tl.arange(0, XBLOCK)   # fast axis for stores (x2, stride-1 in output)
    ty = tl.arange(0, YBLOCK)   # fast axis for loads (y0, stride-1 in input)
    # x_idx: [XBLOCK, 1] — slow axis on loads, fast axis on stores
    x_idx = pid_x * XBLOCK + tx[:, None]
    # y_idx: [1, YBLOCK] — fast axis on loads (y0 stride-1), slow axis on stores
    y_idx = pid_y * YBLOCK + ty[None, :]
    y0_idx = y_idx % 128
    y1_idx = y_idx // 128
    x_mask = x_idx < xnumel
    y_mask = y_idx < ynumel
    mask = x_mask & y_mask
    # COALESCED LOAD: ty (inner dim) varies y0 (stride 1 in input)
    load_addr = y0_idx + 128 * x_idx + 4032000 * y1_idx
    tile_data = tl.load(in_ptr0 + load_addr, mask=mask, other=0.0)
    # COALESCED STORE: tx (outer dim) varies x2 (stride 1 in output)
    store_addr = x_idx + 31500 * y_idx
    tl.store(out_ptr0 + store_addr, tile_data, mask=mask)

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
