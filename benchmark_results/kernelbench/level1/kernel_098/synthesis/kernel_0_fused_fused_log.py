import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_ROWS': 1}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=1),
    ],
    key=['N', 'M'],
)
@triton.jit
def kl_div_kernel(
    pred_ptr,
    target_ptr,
    partial_out_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # Each program handles BLOCK_ROWS rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    
    acc = tl.zeros([BLOCK_ROWS, 1], dtype=tl.float32)
    
    for row_off in tl.static_range(0, BLOCK_ROWS):
        row = row_start + row_off
        row_acc = tl.zeros([1], dtype=tl.float32)
        
        for col_off in range(0, N, BLOCK_N):
            cols = col_off + tl.arange(0, BLOCK_N)
            mask = (cols < N) & (row < M)
            
            pred = tl.load(pred_ptr + row * stride_m + cols, mask=mask, other=1.0)
            target = tl.load(target_ptr + row * stride_m + cols, mask=mask, other=0.0)
            
            log_pred = tl.log(pred)
            
            # xlogy(target, target): target * log(target), with 0 where target==0
            log_target = tl.log(target)
            t_log_t = target * log_target
            is_zero = (target == 0.0)
            is_nan = target != target  # NaN check
            t_log_t = tl.where(is_zero, tl.zeros_like(t_log_t), t_log_t)
            t_log_t = tl.where(is_nan, tl.full(t_log_t.shape, float('nan'), dtype=tl.float32), t_log_t)
            
            # kl term: target * log(target) - target * log(pred)
            kl = t_log_t - target * log_pred
            kl = tl.where(mask, kl, tl.zeros_like(kl))
            
            row_acc += tl.sum(kl, axis=0)
        
        acc[row_off, 0] = row_acc
    
    # Store partial sums (one per row handled by this program)
    for row_off in tl.static_range(0, BLOCK_ROWS):
        row = row_start + row_off
        if row < M:
            tl.store(partial_out_ptr + row, acc[row_off, 0])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
    ],
    key=['M'],
)
@triton.jit
def reduce_and_div_kernel(
    partial_ptr,
    out_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    # Single program reduces all partial sums
    acc = tl.zeros([1], dtype=tl.float32)
    for off in range(0, M, BLOCK_M):
        cols = off + tl.arange(0, BLOCK_M)
        mask = cols < M
        vals = tl.load(partial_ptr + cols, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    result = acc / M.to(tl.float32)
    tl.store(out_ptr, result)


def kernel_function(predictions, targets):
    M, N = predictions.shape
    assert predictions.is_cuda and targets.is_cuda
    assert predictions.dtype == torch.float32 and targets.dtype == torch.float32
    
    partial = torch.empty(M, device=predictions.device, dtype=torch.float32)
    
    BLOCK_ROWS = 1
    grid_rows = (M + BLOCK_ROWS - 1) // BLOCK_ROWS
    
    kl_div_kernel[(grid_rows,)](
        predictions,
        targets,
        partial,
        M,
        N,
        predictions.stride(0),
        BLOCK_ROWS=BLOCK_ROWS,
    )
    
    out = torch.empty((), device=predictions.device, dtype=torch.float32)
    reduce_and_div_kernel[(1,)](
        partial,
        out,
        M,
    )
    
    return out


def get_inputs():
    import torch
    batch_size = 16384
    input_shape = (16384,)
    scale = torch.rand(())
    pred = (torch.rand(batch_size, *input_shape) * scale).softmax(dim=-1).cuda()
    target = torch.rand(batch_size, *input_shape).softmax(dim=-1).cuda()
    return [pred, target]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        # Verify correctness
        pred, target = inputs
        ref = torch.nn.functional.kl_div(torch.log(pred), target, reduction='batchmean')
        print(f"Triton output: {output.item():.6f}")
        print(f"PyTorch ref:   {ref.item():.6f}")
        print(f"Abs diff: {abs(output.item() - ref.item()):.2e}")
        print("Kernel executed successfully.")