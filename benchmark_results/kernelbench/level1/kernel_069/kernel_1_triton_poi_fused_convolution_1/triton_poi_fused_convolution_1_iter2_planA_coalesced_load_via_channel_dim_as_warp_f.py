import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Layout transpose: channels-last (strides 4032000,1,32000,128) -> contiguous NCHW (strides 4032000,31500,250,1)
    # Each CTA handles XBLOCK=128 channels (stride-1 in input = coalesced) for YBLOCK spatial positions
    # Grid: (total_spatial / YBLOCK, 1) where total_spatial = 32*126*250 = 1_008_000
    # xindex covers channels (0..127), yindex covers spatial positions (n,h,w flattened)
    yoffset = tl.program_id(0) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < 1008000
    xoffset = 0
    xindex = tl.arange(0, XBLOCK)[None, :]
    # Decode spatial position from yindex: total_spatial = 32*126*250
    # yindex in [0, 1008000): w = yindex % 250, h = (yindex // 250) % 126, n = yindex // 31500
    w_idx = yindex % 250
    tmp_yw = yindex // 250
    h_idx = tmp_yw % 126
    n_idx = tmp_yw // 126
    # xindex = channel index (0..127), stride-1 in input -> fully coalesced loads
    c_idx = xindex  # shape [1, XBLOCK]
    # Load: in[n,c,h,w] = n*4032000 + c*1 + h*32000 + w*128 (stride-1 along c -> coalesced)
    load_offset = n_idx * 4032000 + c_idx * 1 + h_idx * 32000 + w_idx * 128
    vals = tl.load(in_ptr0 + load_offset, ymask)
    # Store: out[n,c,h,w] = n*4032000 + c*31500 + h*250 + w (strided along c, unavoidable)
    store_offset = n_idx * 4032000 + c_idx * 31500 + h_idx * 250 + w_idx
    tl.store(out_ptr0 + store_offset, vals, ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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
