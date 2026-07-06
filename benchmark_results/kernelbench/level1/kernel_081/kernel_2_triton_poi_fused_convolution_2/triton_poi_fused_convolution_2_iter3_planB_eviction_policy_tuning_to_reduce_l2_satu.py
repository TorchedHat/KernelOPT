import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 3025
    # XBLOCK = HWBLOCK (fast-varying for stores, spatial dim, stride-1 in out_ptr0 NCHW)
    # YBLOCK = CBLOCK  (fast-varying for loads, channel dim, stride-1 in in_ptr0 NHWC)
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    # c_off: channel/batch index (ynumel=24576 = 256*96)
    c_off = pid_c * YBLOCK + tl.arange(0, YBLOCK)
    # hw_off: spatial index (xnumel=3025 = 55*55)
    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)
    hw_mask = hw_off < xnumel
    y0 = c_off % 96   # channel index within batch (0..95)
    y1 = c_off // 96  # batch index (0..255)
    # load_addr shape: [XBLOCK, YBLOCK] = [HWBLOCK, CBLOCK]
    # in_ptr0 is NHWC: addr = y0 + 96*hw_off + 290400*y1
    # YBLOCK (channel) is fast-varying -> coalesced loads
    load_addr = y0[None, :] + 96 * hw_off[:, None] + 290400 * y1[None, :]
    load_mask = hw_mask[:, None]
    tile = tl.load(in_ptr0 + load_addr, load_mask, other=0.0, eviction_policy='evict_first')
    bias = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tile = tile + bias[None, :]
    # store_addr shape: [XBLOCK, YBLOCK] = [HWBLOCK, CBLOCK]
    # out_ptr0 is NCHW: addr = hw_off + 3025*c_off
    # XBLOCK (spatial) is fast-varying -> coalesced stores
    store_addr = hw_off[:, None] + 3025 * c_off[None, :]
    tl.store(out_ptr0 + store_addr, tile, hw_mask[:, None], eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
