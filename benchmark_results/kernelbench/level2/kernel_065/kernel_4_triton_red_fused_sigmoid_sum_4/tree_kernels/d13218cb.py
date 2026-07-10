import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_sigmoid_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 640
    r0_numel = 115520
    X_CHUNK: tl.constexpr = 32  # 32 floats = 128 bytes = one cache line
    pid = tl.program_id(0)      # grid = cdiv(640, 32) = 20 CTAs
    x_base = pid * X_CHUNK
    r0_base = tl.arange(0, R0_BLOCK)
    results = tl.zeros([X_CHUNK], dtype=tl.float32)
    # Process X_CHUNK x-indices sequentially; 1D accumulator keeps register pressure low
    for xi in tl.static_range(X_CHUNK):
        x0 = x_base + xi
        xmask_scalar = x0 < xnumel
        _acc = tl.zeros([R0_BLOCK], dtype=tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = tl.max_contiguous(tl.multiple_of(r0_index, 16), 16)
            tmp0 = tl.load(in_ptr0 + (r0_1 + 115520 * x0),
                           mask=r0_mask & xmask_scalar, other=0.0,
                           eviction_policy='evict_last')
            _acc += tl.where(r0_mask & xmask_scalar, tl.sigmoid(tmp0), 0.0)
        xi_result = tl.sum(_acc, 0)
        # Scatter result into the correct slot of the results vector
        results = tl.where(tl.arange(0, X_CHUNK) == xi, xi_result, results)
    # Single coalesced 128-byte store: 32 contiguous floats
    out_indices = x_base + tl.arange(0, X_CHUNK)
    out_mask = out_indices < xnumel
    tl.store(out_ptr0 + out_indices, results, mask=out_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 640
    r0_numel = 115520
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_sigmoid_sum_4[grid](arg_0, arg_1, 640, 115520)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 95, 95), (577600, 9025, 95, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 5), (5, 1), device='cuda:0', dtype=torch.float32)
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
