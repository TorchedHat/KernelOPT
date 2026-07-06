import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=4),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_m = tl.program_id(0)  # over ynumel=4096 (channel/batch)
    pid_n = tl.program_id(1)  # over xnumel=31500 (spatial)
    m_offsets = pid_m * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    n_offsets = pid_n * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    y0 = m_offsets % 128   # channel index, stride-1 in source
    y1 = m_offsets // 128  # batch index
    x2 = n_offsets         # spatial index
    m_mask = m_offsets < ynumel
    n_mask = n_offsets < xnumel
    load_mask = m_mask[:, None] & n_mask[None, :]
    # Coalesced load: adjacent m-threads differ by 1 in y0 (stride-1 in source)
    src_idx = y0[:, None] + 128 * x2[None, :] + 4032000 * y1[:, None]
    tile = tl.load(in_ptr0 + src_idx, mask=load_mask, other=0.0, eviction_policy='evict_last')
    # Coalesced store: adjacent n-threads differ by 1 in x2 (stride-1 in dest)
    dst_idx = x2[None, :] + 31500 * m_offsets[:, None]
    tl.store(out_ptr0 + dst_idx, tile, mask=load_mask)

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
