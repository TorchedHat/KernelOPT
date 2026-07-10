import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # Existing num_stages=2 configs
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=2, num_stages=2),
        # New num_stages=3 configs
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        # New num_stages=4 configs (smaller tiles to control register pressure)
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # program_id(0) covers ynumel (channel+batch) — FAST dim for coalescing (stride-1 in input)
    # program_id(1) covers xnumel (spatial) — SLOW dim (stride-128 in input)
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK), YBLOCK), YBLOCK)[:, None]   # spatial (stride-128 in input)
    ymask = yindex < xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]   # channel+batch (stride-1 in input)
    xmask = xindex < ynumel
    x0 = (xindex % 128)   # channel within group, stride-1 in input — COALESCED!
    x1 = xindex // 128    # batch index
    y2 = yindex           # spatial index, stride-128 in input
    # Load: consecutive threads (xindex) access x0 with stride-1 -> COALESCED
    tmp0 = tl.load(in_ptr0 + (x0 + 128*y2 + 4032000*x1), xmask & ymask, eviction_policy='evict_last')
    # Transpose register tile: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK] (zero-cost register reorder)
    tmp_t = tl.trans(tmp0)
    # Re-derive store indices with swapped roles for coalesced store:
    xindex_t = xoffset + tl.arange(0, XBLOCK)[:, None]   # slow dim: stride-31500 in output
    yindex_t = yoffset + tl.arange(0, YBLOCK)[None, :]   # fast dim: stride-1 in output -> COALESCED
    xmask_t = xindex_t < ynumel   # ynumel=4096
    ymask_t = yindex_t < xnumel   # xnumel=31500
    # Store: consecutive threads differ in yindex_t (stride-1) -> COALESCED (1 txn/warp)
    tl.store(out_ptr0 + (yindex_t + 31500*xindex_t), tmp_t, xmask_t & ymask_t)

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
