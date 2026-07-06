import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Cross entropy loss fused kernel:
# For each row i:
#   1. Compute max_i = max(predictions[i, :])
#   2. Compute sum_exp_i = sum(exp(predictions[i, :] - max_i))
#   3. log_softmax[i, t] = predictions[i, t] - max_i - log(sum_exp_i)
#   4. loss_i = -log_softmax[i, targets[i]] (if targets[i] != -100, else 0)
# Final: mean(loss_i) over valid samples


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def cross_entropy_per_row_kernel(
    predictions_ptr,
    targets_ptr,
    per_row_loss_ptr,
    per_row_valid_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    One program per row. Two-pass cross entropy with coalesced loads.
    """
    row = tl.program_id(0)

    # Load target for this row
    target = tl.load(targets_ptr + row)

    # Pass 1: compute max over the row
    row_max = tl.full((), float('-inf'), dtype=tl.float32)

    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(predictions_ptr + row * N + cols, mask=mask, other=float('-inf'), eviction_policy='evict_last')
        x = x.to(tl.float32)
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # Pass 2: compute sum of exp(x - max)
    row_sum = tl.zeros((), dtype=tl.float32)

    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(predictions_ptr + row * N + cols, mask=mask, other=float('-inf'), eviction_policy='evict_last')
        x = x.to(tl.float32)
        exp_x = tl.exp(x - row_max)
        exp_x = tl.where(mask, exp_x, 0.0)
        row_sum += tl.sum(exp_x, axis=0)

    log_sum_exp = libdevice.log(row_sum) + row_max

    # Compute loss for this row
    is_valid = target != -100

    # Clamp target to valid range for indexing
    safe_target = tl.where(is_valid, target, tl.zeros_like(target))
    # Handle negative indices
    safe_target = tl.where(safe_target < 0, safe_target + N, safe_target)

    # Load the prediction at the target class
    pred_at_target = tl.load(predictions_ptr + row * N + safe_target).to(tl.float32)

    # log_softmax at target = pred_at_target - log_sum_exp
    log_prob = pred_at_target - log_sum_exp
    loss = -log_prob

    # Zero out loss for invalid targets
    loss = tl.where(is_valid, loss, 0.0)
    valid_int = tl.where(is_valid, 1, 0)

    tl.store(per_row_loss_ptr + row, loss)
    tl.store(per_row_valid_ptr + row, valid_int)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 1024}, num_warps=8),
    ],
    key=['M'],
)
@triton.jit

def reduce_loss_kernel(
    per_row_loss_ptr,
    per_row_valid_ptr,
    output_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    
    total_loss = tl.zeros((), dtype=tl.float32)
    total_valid = tl.zeros((), dtype=tl.float32)
    
    for off in tl.range(0, M, BLOCK_M):
        rows = off + tl.arange(0, BLOCK_M)
        mask = rows < M
        loss = tl.load(per_row_loss_ptr + rows, mask=mask, other=0.0)
        valid = tl.load(per_row_valid_ptr + rows, mask=mask, other=0)
        total_loss += tl.sum(loss, axis=0)
        total_valid += tl.sum(valid.to(tl.float32), axis=0)
    
    mean_loss = total_loss / total_valid
    tl.store(output_ptr, mean_loss)


def kernel_function(predictions, targets):
    M, N = predictions.shape
    
    # Allocate per-row buffers
    per_row_loss = torch.empty(M, device=predictions.device, dtype=torch.float32)
    per_row_valid = torch.empty(M, device=predictions.device, dtype=torch.int32)
    
    # Launch per-row kernel: one program per row
    grid_per_row = (M,)
    cross_entropy_per_row_kernel[grid_per_row](
        predictions,
        targets,
        per_row_loss,
        per_row_valid,
        M,
        N,
    )
    
    # Allocate output scalar
    output = torch.empty((), device=predictions.device, dtype=torch.float32)
    
    # Launch reduction kernel: single program
    grid_reduce = (1,)
    reduce_loss_kernel[grid_reduce](
        per_row_loss,
        per_row_valid,
        output,
        M,
    )
    
    return output


def get_inputs():
    batch_size = 32768
    num_classes = 4096
    predictions = torch.rand(batch_size, num_classes, device='cuda', dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda', dtype=torch.int64)
    return [predictions, targets]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        
        # Reference implementation
        ref_output = torch.nn.functional.cross_entropy(inputs[0], inputs[1])
        
        print(f"Triton output: {output.item()}")
        print(f"PyTorch output: {ref_output.item()}")
        print(f"Max diff: {(output - ref_output).abs().item()}")
        
        assert torch.allclose(output, ref_output, atol=1e-3), f"Mismatch: {output.item()} vs {ref_output.item()}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")