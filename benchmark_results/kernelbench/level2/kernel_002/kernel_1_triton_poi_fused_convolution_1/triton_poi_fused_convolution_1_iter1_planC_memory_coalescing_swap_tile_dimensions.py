import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # pid_x -> channel dim (XBLOCK lanes walk stride-1 in load -> COALESCED)
    chan_offset = tl.program_id(0) * XBLOCK
    chan_index = chan_offset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK]
    chan_mask = chan_index < ynumel
    # pid_y -> spatial dim
    spat_offset = tl.program_id(1) * YBLOCK
    spat_index = spat_offset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    spat_mask = spat_index < xnumel
    mask = chan_mask & spat_mask
    y0 = chan_index % 128   # in-group channel, stride=1 in load -> COALESCED
    y1 = chan_index // 128  # batch index
    y3 = chan_index         # full channel index
    x2 = spat_index         # spatial index
    # LOAD: fast dim is y0 (stride=1) -> 32 consecutive threads = 1 cache line
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask, eviction_policy='evict_last')
    # STORE: x2 is stride-1 in output (out stride along spatial dim = 1)
    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask)

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
