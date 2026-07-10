import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Reinterpret axes:
    #   pid(0) -> spatial pixel blocks (XBLOCK = SPBLOCK)
    #   pid(1) -> batch index (n), YBLOCK is fixed to 128 = CBLOCK
    # ynumel = 4096 = 32*128, xnumel = 31500
    # We tile [CBLOCK=128, SPBLOCK=XBLOCK] over (channel, spatial_pixel) per batch
    pid_sp = tl.program_id(0)   # spatial block index
    pid_n  = tl.program_id(1)   # batch index (one program per batch element)

    sp_off = pid_sp * XBLOCK
    # sp_idx shape: [1, XBLOCK]
    sp_idx = sp_off + tl.arange(0, XBLOCK)[None, :]
    sp_mask = sp_idx < xnumel  # xnumel = sp_numel = 31500

    # c_idx shape: [YBLOCK, 1] = [128, 1]
    c_idx = tl.arange(0, YBLOCK)[:, None]

    n = pid_n  # batch index 0..31

    # LOAD: source layout NHWC stride=(4032000, 1, 32000, 128)
    # Offset simplification: c + 128*sp + 4032000*n (valid because 128*250=32000)
    # Loads are coalesced along C (stride-1 in source)
    src_offset = c_idx + 128 * sp_idx + 4032000 * n  # shape [YBLOCK, XBLOCK]
    vals = tl.load(in_ptr0 + src_offset, mask=sp_mask, other=0.0)

    # STORE: destination layout NCHW stride=(4032000, 31500, 250, 1)
    # Offset: n*4032000 + c*31500 + sp
    # Stores are coalesced along sp/W (stride-1 in NCHW dest)
    dst_offset = n * 4032000 + c_idx * 31500 + sp_idx  # shape [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + dst_offset, vals, mask=sp_mask)

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
