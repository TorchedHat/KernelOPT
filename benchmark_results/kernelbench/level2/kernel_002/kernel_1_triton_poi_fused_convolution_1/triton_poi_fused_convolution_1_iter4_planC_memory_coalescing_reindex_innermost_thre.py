import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32,  'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 128}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 31500 (HW), xnumel = 128 (C), YBLOCK = HWBLOCK, XBLOCK = CBLOCK = 128
    # pid(0) tiles HW, pid(1) tiles N (batch)
    pid_hw = tl.program_id(0)
    pid_n  = tl.program_id(1)
    n = pid_n

    hw_start = pid_hw * YBLOCK
    hw_offs  = hw_start + tl.arange(0, YBLOCK)   # [YBLOCK] = [HWBLOCK]
    c_offs   = tl.arange(0, XBLOCK)              # [XBLOCK] = [CBLOCK=128]

    hw_mask  = hw_offs < ynumel                   # [YBLOCK], ynumel=31500

    # Load tile shape [YBLOCK, XBLOCK] = [HWBLOCK, CBLOCK]:
    # c_offs is innermost (stride=1 in input) -> COALESCED LOAD
    # load index: c + 128*hw + 4032000*n
    load_idx = c_offs[None, :] + 128 * hw_offs[:, None] + 4032000 * n
    tmp = tl.load(in_ptr0 + load_idx, mask=hw_mask[:, None], other=0.0)

    # Transpose tile in registers: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    tmp_T = tl.trans(tmp)

    # Store tile shape [XBLOCK, YBLOCK] = [CBLOCK, HWBLOCK]:
    # hw_offs is innermost (stride=1 in output) -> COALESCED STORE
    # store index: hw + 31500*c + 4032000*n
    store_idx = hw_offs[None, :] + 31500 * c_offs[:, None] + 4032000 * n
    tl.store(out_ptr0 + store_idx, tmp_T, mask=hw_mask[None, :])

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
