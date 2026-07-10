import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: axis-0 = (batch * channel) index, axis-1 = spatial tile index
    # nc_total = 16 * 128 = 2048, spatial_numel = 39304
    nc_idx = tl.program_id(0)   # (batch, channel) flattened: 0..2047
    s_pid  = tl.program_id(1)   # spatial tile index

    n_idx = nc_idx // 128       # batch index [0, 16)
    c_idx = nc_idx % 128        # channel index [0, 128)
    g_idx = c_idx // 16         # group index [0, 8)

    # SCALAR loads: ONE load per CTA — no broadcast waste
    # in_ptr1/in_ptr2: shape (16, 8, 1, 1), strides (8, 1, 128, 128)
    # => flat index = n_idx * 8 + g_idx
    mean_val = tl.load(in_ptr1 + n_idx * 8 + g_idx)
    var_val  = tl.load(in_ptr2 + n_idx * 8 + g_idx)
    # in_ptr3/in_ptr4: shape (128,), stride 1 => index = c_idx
    weight   = tl.load(in_ptr3 + c_idx)
    bias_val = tl.load(in_ptr4 + c_idx)

    # Precompute inv_std once per CTA
    inv_std = libdevice.rsqrt(var_val / 628864.0 + 1.0e-05)

    # Spatial tile: coalesced vector load
    spatial_numel = 39304
    s_offset = s_pid * XBLOCK
    s_range  = s_offset + tl.arange(0, XBLOCK)
    s_mask   = s_range < spatial_numel

    # in_ptr0/out_ptr0: shape (16,128,34,34,34), strides (5030912,39304,1156,34,1)
    flat_idx = n_idx * 5030912 + c_idx * 39304 + s_range

    x_val  = tl.load(in_ptr0 + flat_idx, s_mask, other=0.0)
    # ReLU first, then group norm affine
    x_relu = tl.maximum(x_val, 0.0)
    x_norm = (x_relu - mean_val) * inv_std
    result = x_norm * weight + bias_val
    tl.store(out_ptr0 + flat_idx, result, s_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 80494592
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_native_group_norm_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 80494592)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
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
