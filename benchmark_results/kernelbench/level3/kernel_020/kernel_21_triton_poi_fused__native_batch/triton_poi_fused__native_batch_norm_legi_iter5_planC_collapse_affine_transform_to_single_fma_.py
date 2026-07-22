import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    # Preload all 64-element per-channel vectors into registers ONCE before the loop
    chan_idx   = tl.arange(0, 64)
    chan_mask  = chan_idx < 64
    mean_vec   = tl.load(in_ptr0 + chan_idx, mask=chan_mask)
    var_vec    = tl.load(in_ptr1 + chan_idx, mask=chan_mask)
    weight_vec = tl.load(in_ptr2 + chan_idx, mask=chan_mask)
    bias_vec   = tl.load(in_ptr3 + chan_idx, mask=chan_mask)
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 64)
        tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
        tmp1  = tl.gather(mean_vec,   x0, 0)
        tmp3  = tl.gather(var_vec,    x0, 0)
        tmp12 = tl.gather(weight_vec, x0, 0)
        tmp14 = tl.gather(bias_vec,   x0, 0)
        tmp2 = tmp0 - tmp1
        tmp5 = tmp3 + 1e-05
        tmp6 = libdevice.rsqrt(tmp5)
        tmp11 = tmp2 * tmp6
        tmp15 = tmp11 * tmp12 + tmp14
        tl.store(in_out_ptr0 + xindex, tmp15, xmask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 125440
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 125440, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
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
