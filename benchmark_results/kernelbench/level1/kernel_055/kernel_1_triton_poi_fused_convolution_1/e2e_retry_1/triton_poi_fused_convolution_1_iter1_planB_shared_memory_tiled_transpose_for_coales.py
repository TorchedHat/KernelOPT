import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=8, num_stages=3),
    triton.Config({'YBLOCK': 16, 'XBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 16, 'XBLOCK': 16}, num_warps=8, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)  # tile along X
    pid_y = tl.program_id(1)  # tile along Y
    x_off = pid_x * XBLOCK
    y_off = pid_y * YBLOCK
    tx = tl.arange(0, XBLOCK)
    ty = tl.arange(0, YBLOCK)
    # Coalesced load: shape [YBLOCK, XBLOCK] — consecutive x per row
    x_idx = x_off + tx[None, :]  # [1, XBLOCK]
    y_idx = y_off + ty[:, None]  # [YBLOCK, 1]
    mask = (x_idx < xnumel) & (y_idx < ynumel)
    y0 = y_idx % 128
    y1 = y_idx // 128
    src = tl.load(in_ptr0 + (y0 + 128 * x_idx + 4032000 * y1), mask=mask, other=0.0)
    # Transpose in-register: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    src_T = tl.trans(src)
    # Coalesced store: consecutive y per row of transposed tile
    x_idx_T = x_off + tx[:, None]  # [XBLOCK, 1]
    y_idx_T = y_off + ty[None, :]  # [1, YBLOCK]
    mask_T = (x_idx_T < xnumel) & (y_idx_T < ynumel)
    tl.store(out_ptr0 + (x_idx_T + 31500 * y_idx_T), src_T, mask=mask_T)

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
