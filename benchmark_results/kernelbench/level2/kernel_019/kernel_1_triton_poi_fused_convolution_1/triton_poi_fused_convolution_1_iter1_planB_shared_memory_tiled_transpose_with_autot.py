import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x_base = pid_x * XBLOCK
    y_base = pid_y * YBLOCK

    tx = tl.arange(0, XBLOCK)
    ty = tl.arange(0, YBLOCK)

    # Shape: [YBLOCK, XBLOCK] — ty varies along rows (y), tx varies along cols (x)
    x_idx = x_base + tx[None, :]   # shape [1, XBLOCK] broadcast -> [YBLOCK, XBLOCK]
    y_idx = y_base + ty[:, None]   # shape [YBLOCK, 1] broadcast -> [YBLOCK, XBLOCK]

    # Decompose y_idx for load address
    y0 = y_idx % 128
    y1 = y_idx // 128

    # Load coalesced: consecutive threads in x_idx dimension read consecutive addresses
    load_mask = (x_idx < xnumel) & (y_idx < ynumel)
    src = tl.load(in_ptr0 + (y0 + 128 * x_idx + 4032000 * y1),
                  mask=load_mask, other=0.0, eviction_policy='evict_last')

    # Transpose in-register: shape [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    src_T = tl.trans(src)

    # Build output indices after transpose (XBLOCK == YBLOCK for square tiles)
    x_out = x_base + tx[:, None]   # shape [XBLOCK, 1] broadcast -> [XBLOCK, YBLOCK]
    y_out = y_base + ty[None, :]   # shape [1, YBLOCK] broadcast -> [XBLOCK, YBLOCK]

    # Store coalesced: consecutive threads in y_out dimension write consecutive addresses
    store_mask = (x_out < xnumel) & (y_out < ynumel)
    tl.store(out_ptr0 + (x_out + 31500 * y_out), src_T, mask=store_mask)

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
