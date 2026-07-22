import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    while tile_id < total_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x3 = xindex
        x0 = (xindex % 384)
        x1 = ((xindex // 384) % 196)
        x2 = xindex // 75264
        tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
        x1_mod14 = x1 % 14
        x1_div14 = x1 // 14
        x0_mod32 = x0 % 32
        x0_div32 = x0 // 32
        row_inner = (4 + x1_mod14) % 7
        col_inner = (4 + x1_div14) % 7
        row_outer = (4 + x1_mod14) // 7
        col_outer = (4 + x1_div14) // 7
        ptr0_idx = 32*row_inner + 224*col_inner + 1568*x0_div32 + 18816*row_outer + 56448*col_outer + 169344*x2 + x0_mod32
        ptr1_idx = 7*col_inner + 49*x0_div32 + row_inner
        tmp1 = tl.load(in_ptr0 + ptr0_idx, xmask)
        tmp2 = tl.load(in_ptr1 + ptr1_idx, xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x3), xmask)
        tmp_inner = libdevice.fma(1.0, tmp5, tmp6)
        tmp_mid   = libdevice.fma(1.0, tmp2, tmp_inner)
        tmp_mid2  = libdevice.fma(1.0, tmp1, tmp_mid)
        tmp8      = libdevice.fma(1.0, tmp0, tmp_mid2)
        tl.store(in_out_ptr0 + (x3), tmp8, xmask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 752640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 752640, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((90, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
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
