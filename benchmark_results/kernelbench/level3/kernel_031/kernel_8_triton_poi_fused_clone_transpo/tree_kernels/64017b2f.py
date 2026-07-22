import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiled transpose: input (8, 16384, 32) -> output (16384, 8, 32)
    # XBLOCK covers the N=16384 dimension (pid axis 0)
    # BLOCK_M=8 covers the M=8 dimension (pid axis 1), always fits in 1 block
    BLOCK_M: tl.constexpr = 8
    pid_n = tl.program_id(0)   # covers N=16384 dimension
    pid_m = tl.program_id(1)   # covers M=8 dimension
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offs = pid_n * XBLOCK + tl.arange(0, XBLOCK)    # [XBLOCK]
    m_mask = m_offs < 8
    n_mask = n_offs < 16384
    # stride_im=524288, stride_in=32, stride_ik=1 (input shape 8,16384,32)
    # stride_om=256, stride_on=32, stride_ok=1 (output shape 16384,8,32)
    for k in tl.static_range(32):  # unroll K=32 at compile time
        ptrs_in = in_ptr0 + (
            m_offs[:, None] * 524288 +
            n_offs[None, :] * 32 +
            k
        )  # [BLOCK_M, XBLOCK], coalesced over n (stride_in=32)
        tile = tl.load(ptrs_in, mask=m_mask[:, None] & n_mask[None, :])
        ptrs_out = out_ptr0 + (
            n_offs[:, None] * 256 +
            m_offs[None, :] * 32 +
            k
        )  # [XBLOCK, BLOCK_M], coalesced over n (stride_om=256)
        tl.store(ptrs_out, tl.trans(tile), mask=n_mask[:, None] & m_mask[None, :])

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 8, 32), (256, 32, 1), device='cuda:0', dtype=torch.float32)
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
