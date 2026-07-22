import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    # 2D decomposition: outer = spatial positions (xnumel/6 = 8388608), inner = C=8 (power-of-2 >= 6)
    C: tl.constexpr = 8
    spatial_numel = 8388608  # xnumel / 6 = 50331648 / 6
    sp_offset = tl.program_id(0) * XBLOCK
    sp_index = sp_offset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    c_index  = tl.arange(0, C)[None, :]                   # [1, C=8]
    flat_index = sp_index * 6 + c_index                    # [XBLOCK, C], stride-1 in C dim
    sp_mask = sp_index < spatial_numel                     # [XBLOCK, 1]
    c_mask  = c_index < 6                                  # [1, C]
    mask = sp_mask & c_mask                                # [XBLOCK, C]
    tmp0 = tl.load(in_ptr0 + flat_index, mask, eviction_policy='evict_first')
    tmp2 = tl.maximum(tl.zeros([XBLOCK, C], dtype=tl.float32), tmp0)
    tl.store(out_ptr0 + flat_index, tmp2, mask, eviction_policy='evict_first')
    tl.store(out_ptr1 + flat_index, tmp2, mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_relu_2[grid](arg_0, arg_1, arg_2, 50331648)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8388608, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 6, 256, 256), (393216, 1, 1536, 6), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 6, 256, 256), (393216, 1, 1536, 6), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
