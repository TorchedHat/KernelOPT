import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 8}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    # LOAD: in_ptr0 strides (4032000, 1, 32000, 128) -> y0 is stride-1
    # Apply vectorization hints: y0 varies along YBLOCK (rows), stride-1
    load_idx = y0 + 128 * x2 + 4032000 * y1
    load_idx = tl.max_contiguous(tl.multiple_of(load_idx, [128, 1]), [128, 1])
    tmp0 = tl.load(in_ptr0 + load_idx, mask=xmask, eviction_policy='evict_last')
    # STORE: out_ptr0 strides (4032000, 31500, 250, 1) -> x2 is stride-1
    store_idx = x2 + 31500 * y3
    store_idx = tl.max_contiguous(tl.multiple_of(store_idx, [1, 128]), [1, 128])
    tl.store(out_ptr0 + store_idx, tmp0, mask=xmask)

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
