import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    # xindex covers channels (0..31), yindex covers spatial*batch
    y0 = xindex          # channel index, shape [1, XBLOCK]
    x2 = yindex % (xnumel // 32)  # spatial index (0..65535), shape [YBLOCK, 1]
    y1 = yindex // (xnumel // 32)  # batch index (0..127), shape [YBLOCK, 1]
    # Load BN params indexed by channel (small, reused -> evict_last)
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + y0, None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + y0, None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + y0, None, eviction_policy='evict_last')
    # Load input NCHW: stride=(2097152, 65536, 256, 1)
    # offset = spatial(x2) + 256*256*channel(y0) + 256*256*32*batch(y1)
    tmp0 = tl.load(in_ptr0 + (x2 + 65536*y0 + 2097152*y1), None, eviction_policy='evict_first')
    # BN + ReLU
    tmp_rsqrt = tl.math.rsqrt(tmp3 + 1e-5)
    tmp_norm = (tmp0 - tmp1) * tmp_rsqrt * tmp12 + tmp14
    tmp_relu = tl.maximum(tl.zeros([YBLOCK, XBLOCK], tl.float32), tmp_norm)
    # Store NHWC: stride=(2097152, 1, 8192, 32)
    # offset = channel(y0) + 32*spatial(x2) + 2097152*batch(y1)
    # Consecutive x-threads differ by 1 in y0 -> COALESCED store
    tl.store(out_ptr0 + (y0 + 32*x2 + 2097152*y1), tmp_relu, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    ynumel = 4096
    xnumel = 65536
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4096, 65536, YBLOCK=8, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((128, 32, 256, 256), (2097152, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 32, 256, 256), (2097152, 1, 8192, 32), device='cuda:0', dtype=torch.float32)
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
