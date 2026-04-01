import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L,  # Logsumexp for backward pass
    Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    n_heads,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Identify which block we are processing
    pid = tl.program_id(0)
    grid_m = tl.cdiv(N_CTX, BLOCK_M)
    
    # Batch and Head indexing
    off_hz = tl.program_id(1)
    off_b = off_hz // n_heads
    off_h = off_hz % n_heads

    # Offset pointers for Q, K, V
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    # Initialize pointers to the start of the specific block
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block - stays in SRAM for the duration of the inner loop
    q_ptr = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptr)

    # Online Softmax statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Scale Q early
    q = (q * sm_scale).to(tl.float16)

    # Iterate over the K, V sequence (the "Inner" loop)
    # This is the Tiling part of Flash Attention
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K and V blocks
        k_ptr = K + k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk) + start_n * stride_kn
        v_ptr = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk) + start_n * stride_vn
        
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)

        # Dot product: Attention Scores (S = QK^T)
        qk = tl.dot(q, tl.trans(k))

        # Online Softmax: update running max and sum of exponentials
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Rescale the accumulator to maintain the moving average
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)
        
        acc = acc * alpha[:, None]
        # Update accumulator with (P * V)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)) * beta[:, None]

        # Update statistics
        l_i = l_i * alpha + l_ij * beta
        m_i = m_next

    # Final normalization of the attention output
    acc = acc / l_i[:, None]
    
    # Store L (logsumexp) for potential backward pass use
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i))

    # Store result to global memory
    out_ptr = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptr, acc.to(Out.dtype.element_ty))


def run_flash_attention(q, k, v):
    batch, n_heads, seq_len, d_model = q.shape
    sm_scale = 1.0 / (d_model**0.5)
    
    out = torch.empty_like(q)
    # L stores logsumexp for each row (needed for backprop)
    L = torch.empty((batch * n_heads, seq_len), device=q.device, dtype=torch.float32)

    # OPTIMIZED: increase_eligible_warps_per_scheduler
    # Original: BLOCK_M = 128
    # Standard tiling sizes for generic 16/32GB GPUs
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, L, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=d_model,
        num_warps=4,
        num_stages=2,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--s', type=int, default=1024)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()

    # Flash attention is typically run in FP16 or BF16
    dtype = torch.float16
    Q = torch.randn((args.b, args.h, args.s, args.d), device='cuda', dtype=dtype)
    K = torch.randn((args.b, args.h, args.s, args.d), device='cuda', dtype=dtype)
    V = torch.randn((args.b, args.h, args.s, args.d), device='cuda', dtype=dtype)

    if args.profile_only:
        # Simple warmup and run for profiling
        for _ in range(10):
            _ = run_flash_attention(Q, K, V)
        torch.cuda.synchronize()
        print(f"Profiled standard Flash Attention: S={args.s}, D={args.d}")
    else:
        # Correctness check against PyTorch's native Fused Attention
        Y_triton = run_flash_attention(Q, K, V)
        Y_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        if torch.allclose(Y_triton, Y_torch, atol=1e-2, rtol=1e-2):
            print("✓ Correctness passed")
        else:
            max_err = (Y_triton - Y_torch).abs().max().item()
            print(f"✗ Correctness FAILED: max_err={max_err}")


if __name__ == '__main__':
    main()