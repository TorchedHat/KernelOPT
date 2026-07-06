import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Reinterpret: XBLOCK tiles channel dim (cnumel=128, stride-1 in source → coalesced load)
    #              YBLOCK tiles spatial dim (snumel=31500, stride-1 in dest → coalesced store)
    # ynumel=128 (channel), xnumel=31500 (spatial) — roles swapped vs original
    cnumel = 128
    snumel = 31500

    b = tl.program_id(2)   # batch index [0, 32)

    # XBLOCK = channel fast dim (stride-1 in source NCHW)
    coffset = tl.program_id(0) * XBLOCK
    cindex = coffset + tl.arange(0, XBLOCK)[None, :]
    cmask = cindex < cnumel

    # YBLOCK = spatial slow dim (stride-1 in dest NHWC)
    soffset = tl.program_id(1) * YBLOCK
    sindex = soffset + tl.arange(0, YBLOCK)[:, None]
    smask = sindex < snumel

    mask = smask & cmask

    # LOAD: in_ptr0 strides (4032000,1,32000,128) for (N,C,H,W)
    # flat index = b*4032000 + c + s*128
    # cindex varies along warp fast dim with stride 1 → COALESCED LOAD
    tmp0 = tl.load(in_ptr0 + (b * 4032000 + cindex + sindex * 128), mask=mask, other=0.0, eviction_policy='evict_first')

    # STORE: out_ptr0 strides (4032000,31500,250,1) for (N,C,H,W) → NHWC layout
    # flat index = b*4032000 + c*31500 + s
    # sindex varies along warp slow dim with stride 1 → COALESCED STORE
    tl.store(out_ptr0 + (b * 4032000 + cindex * 31500 + sindex), tmp0, mask=mask, eviction_policy='evict_first')

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
