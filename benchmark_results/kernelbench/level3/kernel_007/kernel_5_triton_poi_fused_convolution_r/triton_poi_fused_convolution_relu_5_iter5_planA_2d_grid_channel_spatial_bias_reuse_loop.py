import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    # XBLOCK=256: covers full channel dim (C=256) per block
    # Each block loops over SBLOCK=4 spatial positions, loading bias once
    SBLOCK: tl.constexpr = 4
    spatial_pid = tl.program_id(0)
    c_idx = tl.arange(0, XBLOCK)  # shape [256], covers full channel dim
    bias = tl.load(in_ptr0 + c_idx)  # load all 256 bias values once per block
    spatial_base = spatial_pid * SBLOCK
    for s in tl.static_range(SBLOCK):
        spatial_idx = spatial_base + s
        offsets = spatial_idx * XBLOCK + c_idx
        mask = spatial_idx < 746496
        data = tl.load(in_out_ptr0 + offsets, mask=mask, eviction_policy='evict_first')
        result = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), data + bias)
        tl.store(in_out_ptr0 + offsets, result, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 191102976
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_relu_5[grid](arg_0, arg_1, 191102976, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 256, 27, 27), (186624, 1, 6912, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
