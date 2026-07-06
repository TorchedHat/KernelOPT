import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4,  'YBLOCK': 256},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 512},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 512},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},   num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 1024}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 501760   # N * H * W = 10 * 224 * 224
    xnumel = 3        # C = 3 (channels)
    yoffset = tl.program_id(1) * YBLOCK
    yindex  = yoffset + tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1]
    ymask   = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[None, :]   # [1, XBLOCK]
    xmask   = xindex < xnumel
    # Decompose y into batch and spatial indices
    y_hw = yindex % 50176   # spatial pixel index (0..50175)
    y_n  = yindex // 50176  # batch index (0..9)
    # Load: NCHW flat index = y_n*150528 + xindex*50176 + y_hw
    load_idx = y_n * 150528 + xindex * 50176 + y_hw
    tmp0 = tl.load(in_ptr0 + load_idx, xmask & ymask)
    # Store: NHWC flat index = y_n*150528 + y_hw*3 + xindex  [stride-1 in x -> COALESCED]
    store_idx = y_n * 150528 + y_hw * 3 + xindex
    tl.store(out_ptr0 + store_idx, tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
