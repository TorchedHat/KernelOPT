import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': ty, 'XBLOCK': tx}, num_warps=nw, num_stages=ns)
        for ty in [16, 32, 64]
        for tx in [16, 32, 64]
        for nw in [4, 8]
        for ns in [2, 3]
        if ty * tx <= 4096
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # YBLOCK threads walk the y (channel) dimension -- stride=1 in source for coalesced loads
    ys = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    xs = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    ymask = ys < ynumel
    xmask = xs < xnumel
    y0 = ys % 128   # channel -- stride=1 in source
    y1 = ys // 128  # batch
    load_mask = ymask[:, None] & xmask[None, :]
    # Coalesced: YBLOCK threads walk y0 (stride=1), XBLOCK threads walk xs (stride=128)
    src_offsets = y0[:, None] + 128 * xs[None, :] + 4032000 * y1[:, None]
    tile = tl.load(in_ptr0 + src_offsets, mask=load_mask, eviction_policy='evict_last')
    dst_offsets = xs[None, :] + 31500 * ys[:, None]
    tl.store(out_ptr0 + dst_offsets, tile, mask=load_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
