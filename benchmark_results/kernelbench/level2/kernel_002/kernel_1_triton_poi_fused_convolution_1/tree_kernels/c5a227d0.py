import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 256}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    C = 128
    N_STRIDE = 4032000
    C_OUT_STRIDE = 31500
    HW_IN_STRIDE = 128
    hw_off = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    hw_mask = hw_off < xnumel
    nc_off = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    nc_mask = nc_off < ynumel
    n_idx = nc_off // C
    c_idx = nc_off % C
    load_idx = (n_idx[None, :] * N_STRIDE
                + c_idx[None, :] * 1
                + hw_off[:, None] * HW_IN_STRIDE)
    mask2d = hw_mask[:, None] & nc_mask[None, :]
    vals = tl.load(in_ptr0 + load_idx, mask=mask2d)
    store_idx = (n_idx[None, :] * N_STRIDE
                 + c_idx[None, :] * C_OUT_STRIDE
                 + hw_off[:, None] * 1)
    tl.store(out_ptr0 + store_idx, vals, mask=mask2d)

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
