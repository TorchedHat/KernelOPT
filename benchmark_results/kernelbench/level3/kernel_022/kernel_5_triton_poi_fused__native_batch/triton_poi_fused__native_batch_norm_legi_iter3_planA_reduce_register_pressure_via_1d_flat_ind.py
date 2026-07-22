import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # numel = ynumel * xnumel = 1920 * 12544 = 24084480
    # Use 1D flat indexing over all elements to avoid 2D broadcast register pressure
    numel = 24084480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)   # [XBLOCK] — 1D, NO broadcast
    xmask = xindex < numel
    # Decompose flat index into (channel, spatial, batch)
    channel = xindex % 192
    tmp_bc  = xindex // 192
    spatial = tmp_bc % 12544
    batch   = tmp_bc // 12544
    # Load per-channel BN params (stride-1)
    tmp_mean   = tl.load(in_ptr1 + channel, mask=xmask, eviction_policy='evict_last')
    tmp_var    = tl.load(in_ptr2 + channel, mask=xmask, eviction_policy='evict_last')
    tmp_weight = tl.load(in_ptr3 + channel, mask=xmask, eviction_policy='evict_last')
    tmp_bias   = tl.load(in_ptr4 + channel, mask=xmask, eviction_policy='evict_last')
    # Precompute per-channel scale and bias
    tmp_eps     = tl.full([1], 1e-05, tl.float32)
    tmp_inv_std = libdevice.rsqrt(tmp_var + tmp_eps)        # rsqrt(var + eps)
    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)
    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale
    # Load main input: in_ptr0 shape (125440,192) stride (192,1)
    tmp0 = tl.load(in_ptr0 + (channel + 192 * (spatial + 12544 * batch)), mask=xmask, eviction_policy='evict_first')
    # BN transform
    tmp15 = tmp0 * combined_scale + combined_bias
    # Store: out_ptr0 shape (10,192,112,112) stride (2408448,12544,112,1)
    tl.store(out_ptr0 + (spatial + 12544 * channel + 2408448 * batch), tmp15, mask=xmask)

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
