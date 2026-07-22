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
        triton.Config({'XBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 64}, num_warps=8),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=2),
        triton.Config({'XBLOCK': 64}, num_warps=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_222(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # 2D grid: pid(0) = channel tile, pid(1) = spatial tile
    # XBLOCK = CBLOCK (channel tile size), SBLOCK fixed at 32
    SBLOCK: tl.constexpr = 32
    C: tl.constexpr = 1760
    S: tl.constexpr = 490
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    c_off = pid_c * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK]
    s_off = pid_s * SBLOCK + tl.arange(0, SBLOCK)   # [SBLOCK]
    c_mask = c_off < C
    s_mask = s_off < S
    c2d = c_off[None, :]   # [1, XBLOCK]
    s2d = s_off[:, None]   # [SBLOCK, 1]
    mask2d = s_mask[:, None] & c_mask[None, :]
    # Reconstruct strided flat offset for in_ptr0 (strides: 86240, 1, 12320, 1760)
    # s_off encodes (batch, h, w): batch = s_off // 49, hw = s_off % 49
    # h = hw // 7, w = hw % 7
    # flat = batch*86240 + c*1 + h*12320 + w*1760
    batch = s2d // 49
    hw    = s2d % 49
    h     = hw // 7
    w     = hw % 7
    flat_idx = batch * 86240 + c2d + h * 12320 + w * 1760  # [SBLOCK, XBLOCK]
    tmp0  = tl.load(in_ptr0 + flat_idx, mask2d, eviction_policy='evict_first')
    tmp1  = tl.load(in_ptr1 + c2d, mask=c_mask[None, :], eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr2 + c2d, mask=c_mask[None, :], eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + c2d, mask=c_mask[None, :], eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + c2d, mask=c_mask[None, :], eviction_policy='evict_last')
    tmp2  = tmp0 - tmp1
    tmp5  = tmp3 + tl.full([1, 1], 1e-05, tl.float32)
    tmp6  = tl.sqrt_rn(tmp5)
    tmp10 = 1.0 / tmp6
    tmp11 = tmp2 * tmp10
    tmp15 = tmp11 * tmp12 + tmp14
    tmp17 = tl.maximum(tl.full([1, 1], 0, tl.int32), tmp15)
    tl.store(out_ptr0 + flat_idx, tmp17, mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 862400
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_222[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 862400)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 1760, 7, 7), (86240, 1, 12320, 1760), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1760,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1760,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1760,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1760,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1760, 7, 7), (86240, 1, 12320, 1760), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
