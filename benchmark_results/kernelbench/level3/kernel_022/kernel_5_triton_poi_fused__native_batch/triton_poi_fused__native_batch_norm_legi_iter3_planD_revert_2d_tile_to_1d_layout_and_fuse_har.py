import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256, 'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 256}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},   num_warps=2, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 12544
    # YBLOCK covers channel+batch dimension (ynumel=1920)
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape [YBLOCK, 1]
    ymask = yindex < ynumel
    # XBLOCK covers spatial dimension (xnumel=12544)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape [1, XBLOCK]
    xmask = xindex < xnumel
    # Index decomposition
    y0 = yindex % 192       # channel index [0, 192), shape [YBLOCK, 1]
    y1 = yindex // 192      # batch index [0, 10), shape [YBLOCK, 1]
    x2 = xindex             # spatial index [0, 12544), shape [1, XBLOCK]
    y3 = yindex             # channel+batch combined index, shape [YBLOCK, 1]
    # Channel parameter loads: indexed by y0 (channel only)
    tmp_mean   = tl.load(in_ptr1 + y0, mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
    tmp_var    = tl.load(in_ptr2 + y0, mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
    tmp_weight = tl.load(in_ptr3 + y0, mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
    tmp_bias   = tl.load(in_ptr4 + y0, mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
    # Precompute per-channel scale and bias
    tmp_inv_std = libdevice.rsqrt(tmp_var + 1e-05)          # rsqrt(var + eps)
    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)  [YBLOCK, 1]
    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale        [YBLOCK, 1]
    # Load main input: address formula y0 + 192*x2 + 2408448*y1
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 2408448*y1), mask=ymask & xmask, eviction_policy='evict_first')
    # Batch norm: FMA broadcasts [YBLOCK,1] over [1,XBLOCK]
    tmp15 = tmp0 * combined_scale + combined_bias
    # Fused hardtanh clamp [-1.0, 1.0]
    tmp15 = tl.clamp(tmp15, -1.0, 1.0)
    # Store: address formula x2 + 12544*y3
    tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, mask=ymask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    ynumel = 1920
    xnumel = 12544
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((125440, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 192, 112, 112), (2408448, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
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
