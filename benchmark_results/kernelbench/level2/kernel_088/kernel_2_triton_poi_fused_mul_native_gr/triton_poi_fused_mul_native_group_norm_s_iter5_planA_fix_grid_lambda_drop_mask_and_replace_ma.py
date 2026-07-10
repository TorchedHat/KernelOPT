import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_mul_native_group_norm_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x2 = xindex
    x0 = (xindex % 8192)
    tmp0 = tl.load(in_out_ptr0 + (x2))
    tmp1 = tl.load(in_ptr0 + (x2 // 32), eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 // 32), eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp_var_norm = tmp3 * 0.03125 + 1e-05
    tmp8 = libdevice.rsqrt(tmp_var_norm)
    tmp_xnorm = tmp2 * tmp8
    tmp_affine = tmp_xnorm * tmp10 + tmp12
    tmp_silu1 = tmp_affine * tl.sigmoid(tmp_affine)
    tmp_scaled = tmp_silu1 * tmp16
    tmp_silu2 = tmp_scaled * tl.sigmoid(tmp_scaled)
    tl.store(in_out_ptr0 + (x2), tmp_silu2)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_mul_native_group_norm_sigmoid_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8388608)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 256, 1, 1), (256, 1, 262144, 262144), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 256, 1, 1), (256, 1, 262144, 262144), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
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
