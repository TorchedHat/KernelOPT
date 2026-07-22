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
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    # --- CTA-level preload of 192-element channel vectors (before main computation) ---
    # tl.arange requires power-of-2; use 256 (next power of 2 >= 192) with mask
    chan_idx  = tl.arange(0, 256)
    chan_mask = chan_idx < 192
    ch_mean   = tl.load(in_ptr0 + chan_idx, mask=chan_mask, other=0.0)
    ch_var    = tl.load(in_ptr1 + chan_idx, mask=chan_mask, other=0.0)
    ch_weight = tl.load(in_ptr2 + chan_idx, mask=chan_mask, other=0.0)
    ch_bias   = tl.load(in_ptr3 + chan_idx, mask=chan_mask, other=0.0)
    # --- Main per-element computation ---
    xnumel = 94080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0  = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1  = tl.gather(ch_mean,   x0, 0)
    tmp3  = tl.gather(ch_var,    x0, 0)
    tmp12 = tl.gather(ch_weight, x0, 0)
    tmp14 = tl.gather(ch_bias,   x0, 0)
    tmp2  = tmp0 - tmp1
    tmp4  = tl.full([1], 1e-05, tl.float32)
    tmp5  = tmp3 + tmp4
    tmp6  = tl.sqrt_rn(tmp5)
    tmp10 = 1.0 / tmp6
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.maximum(tmp15, 0.0)
    tmp17 = tl.minimum(tmp16, 6.0)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 94080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 94080)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
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
