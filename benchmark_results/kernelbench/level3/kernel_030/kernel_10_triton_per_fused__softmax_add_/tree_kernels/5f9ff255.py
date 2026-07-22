import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_add_clamp_clone_exp_index_log_mul_permute_prepare_softmax_online_sigmoid_sub_unsqueeze_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 94080
    r0_numel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # [XBLOCK] -- 1D
    xmask = xindex < xnumel
    x1 = ((xindex // 49) % 3)   # [XBLOCK]
    x0 = (xindex % 49)           # [XBLOCK]
    x5 = xindex // 49            # [XBLOCK]

    # Load per-x scale (broadcast scalar per x)
    scale = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')  # [XBLOCK]
    scale_exp = tl.exp(tl.minimum(scale, 4.605170249938965))             # [XBLOCK]

    # Online softmax pass 1: compute running max m_i and normalizer d_i
    m_i = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)
    d_i = tl.zeros([XBLOCK], dtype=tl.float32)
    for r0 in tl.static_range(49):
        score = tl.load(in_ptr0 + (r0 + 49*xindex), xmask, other=0.0)   # [XBLOCK]
        idx_raw = tl.load(in_ptr2 + (r0 + 49*x0), xmask, other=0)        # [XBLOCK]
        idx = tl.where(idx_raw < 0, idx_raw + 169, idx_raw)               # [XBLOCK]
        bias = tl.load(in_ptr3 + (x1 + 3*idx), xmask, eviction_policy='evict_last')  # [XBLOCK]
        logit = score * scale_exp + tl.sigmoid(bias) * 16.0               # [XBLOCK]
        m_new = tl.maximum(m_i, logit)
        d_i = d_i * tl.exp(m_i - m_new) + tl.exp(logit - m_new)
        m_i = m_new

    # Online softmax pass 2: normalize and store
    for r0 in tl.static_range(49):
        score = tl.load(in_ptr0 + (r0 + 49*xindex), xmask, other=0.0)
        idx_raw = tl.load(in_ptr2 + (r0 + 49*x0), xmask, other=0)
        idx = tl.where(idx_raw < 0, idx_raw + 169, idx_raw)
        bias = tl.load(in_ptr3 + (x1 + 3*idx), xmask, eviction_policy='evict_last')
        logit = score * scale_exp + tl.sigmoid(bias) * 16.0
        softmax_val = tl.exp(logit - m_i) / d_i                           # [XBLOCK]
        tl.store(out_ptr2 + (r0 + 49*x0 + 2432*x5), softmax_val, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 94080
    r0_numel = 49
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_add_clamp_clone_exp_index_log_mul_permute_prepare_softmax_online_sigmoid_sub_unsqueeze_view_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 94080, 49, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((1920, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((169, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((640, 3, 49, 49), (7296, 2432, 49, 1), device='cuda:0', dtype=torch.float32)
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
