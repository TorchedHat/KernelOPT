import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 8},  num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_add_addmm_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # xnumel here is ynumel = 65536 (number of rows = 128*512)
    # XBLOCK here is YBLOCK (number of rows per CTA)
    HIDDEN: tl.constexpr = 768
    HIDDEN_POW2: tl.constexpr = 1024  # next power of 2 >= 768

    # hidden dimension — use power-of-2 arange with mask for 768-wide bias
    h0 = tl.arange(0, HIDDEN_POW2)    # [1024]
    hmask = h0 < HIDDEN                # [1024], mask out h0 >= 768

    # row (batch*seq) dimension
    yoffset = tl.program_id(0) * XBLOCK
    y0 = yoffset + tl.arange(0, XBLOCK)  # [XBLOCK]
    ymask = y0 < xnumel

    # Load bias vectors ONCE per CTA — reused across all XBLOCK rows
    bias1 = tl.load(in_ptr1 + h0, hmask, other=0.0)   # [1024], stays in L1
    bias2 = tl.load(in_ptr2 + h0, hmask, other=0.0)   # [1024], stays in L1

    # 2D index into row-major [65536, 768] tensors
    idx = y0[:, None] * HIDDEN + h0[None, :]   # [XBLOCK, 1024]
    mask2d = ymask[:, None] & hmask[None, :]   # [XBLOCK, 1024]

    tmp0 = tl.load(in_ptr0     + idx, mask2d, other=0.0, eviction_policy='evict_first')
    tmp2 = tl.load(in_out_ptr0 + idx, mask2d, other=0.0, eviction_policy='evict_first')
    tmp6 = tl.load(in_ptr3     + idx, mask2d, other=0.0, eviction_policy='evict_first')

    # bias broadcast [1024] -> [XBLOCK, 1024]
    tmp3 = bias1[None, :] + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = bias2[None, :] + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + idx, tmp8, mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_addmm_view_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 50331648)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
