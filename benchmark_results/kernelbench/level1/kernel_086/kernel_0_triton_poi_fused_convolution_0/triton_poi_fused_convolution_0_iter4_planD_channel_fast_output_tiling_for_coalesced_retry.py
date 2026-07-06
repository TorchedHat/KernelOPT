import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 256}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 512}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    # XBLOCK tiles the CHANNEL*BATCH dim (fast output axis, stride=1 in NHWC)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK] fast
    xmask = xindex < ynumel  # guard against xindex >= 30
    # YBLOCK tiles the SPATIAL dim (slow output axis, stride=3 in NHWC)
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1] slow
    ymask = yindex < xnumel  # guard against yindex >= 50176
    # Decompose xindex into channel c and batch n
    c = xindex % 3   # [1, XBLOCK] channel index (0..2)
    n = xindex // 3  # [1, XBLOCK] batch index (0..9)
    # Load from NCHW input: strides (150528, 50176, 224, 1)
    # NCHW offset = n*150528 + c*50176 + spatial_pixel
    load_idx = n * 150528 + c * 50176 + yindex  # [YBLOCK, XBLOCK] broadcast
    val = tl.load(in_ptr0 + load_idx, mask=xmask & ymask, other=0.0,
                  eviction_policy='evict_first')
    # Store to NHWC output: strides (150528, 1, 672, 3)
    # NHWC offset = n*150528 + spatial_pixel*3 + c
    # c is the FAST axis (stride=1) -> consecutive threads write consecutive addrs
    store_idx = n * 150528 + yindex * 3 + c  # [YBLOCK, XBLOCK] broadcast
    tl.store(out_ptr0 + store_idx, val, mask=xmask & ymask)

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
