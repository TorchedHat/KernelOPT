import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=4),
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # pid_0 = channel index (0..111), pid_1 = spatial tile
    c = tl.program_id(0)          # channel index, no modulo needed
    spatial_offset = tl.program_id(1) * XBLOCK
    spatial_idx = spatial_offset + tl.arange(0, XBLOCK)[:]
    spatial_mask = spatial_idx < 1960  # 10 * 14 * 14
    # Tensor strides: (10,112,14,14) with strides (21952,1,1568,112)
    # flat_idx = spatial_idx * 112 + c  (channel is stride-1)
    flat_idx = spatial_idx * 112 + c
    # Channel scalars: scalar load, no mask needed (c always < 112)
    tmp1  = tl.load(in_ptr0 + c)   # mean[c]
    tmp3  = tl.load(in_ptr1 + c)   # var[c]
    tmp12 = tl.load(in_ptr2 + c)   # weight[c]
    tmp14 = tl.load(in_ptr3 + c)   # bias[c]
    tmp0 = tl.load(in_out_ptr0 + flat_idx, spatial_mask)
    tmp_var_eps = tmp3 + tl.full([1], 1e-05, tl.float32)
    tmp_inv_std = tl.math.rsqrt(tmp_var_eps)
    tmp_normed = (tmp0 - tmp1) * tmp_inv_std
    tmp_result = tl.fma(tmp_normed, tmp12, tmp14)
    tl.store(in_out_ptr0 + flat_idx, tmp_result, spatial_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 219520
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
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
