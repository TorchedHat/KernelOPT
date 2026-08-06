# Kernel Optimization Log

## Summary

- **Kernel**: `reverse_cumsum_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 40420 ns — **Best latency**: 40420 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `reverse_cumsum_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_091/synthesis/kernel_0_fused_fused_cumsum.py`
- **Framework**: triton
- **Launch**: Block (1024,1,1), Grid (4,1,1), Waves/SM: 0.03
- **Registers/thread**: 64
- **Duration**: 40420 ns
- **SM Throughput**: 1.79 % | **Memory**: 1.50 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 49.92 %
- **IPC Active**: 2.41 inst/cycle | **No Eligible**: 39.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_parallelism_with_row_batching_xblock | 0.959× | 0.997× | ✓ |  |
| B | restructure_grid_2d_and_eliminate_double_load | 0.947× | 14.333× | ✓ |  |
| C | eliminate_redundant_reverse_load_vectorize_memory_access | 0.924× | 0.935× | ✓ |  |
| D | parallelize_within_row_and_reduce_register_pressure | 0.883× | 1.063× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_grid_par... | Δ | restructure_grid_... | Δ | eliminate_redunda... | Δ | parallelize_withi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 40420 ns | 40540 ns | +120 ns (+0.3%) | 2820 ns | -37600 ns (-93.0%) | 43230 ns | +2810 ns (+7.0%) | 38020 ns | -2400 ns (-5.9%) |
| SM Throughput (%) | 1.79 % | 1.78 % | -0.01 pp | 0.16 % | -1.63 pp | 1.66 % | -0.13 pp | 1.90 % | +0.11 pp |
| Memory Throughput (%) | 1.50 % | 1.50 % | +0.00 pp | 0.42 % | -1.08 pp | 1.58 % | +0.08 pp | 1.53 % | +0.03 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.05 % | +0.01 pp | 0.04 % | +0.00 pp | 0.05 % | +0.01 pp |
| Achieved Occupancy (%) | 49.92 % | 49.93 % | +0.01 pp | 6.24 % | -43.68 pp | 49.91 % | -0.01 pp | 49.93 % | +0.01 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 47.00 register/thread | -17 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.00 | -0.03 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 39.50 % | 39.64 % | +0.14 pp | 89.28 % | +49.78 pp | 42.71 % | +3.21 pp | 37.06 % | -2.44 pp |
| IPC Active | 2.41 inst/cycle | 2.39 inst/cycle | -0.02 | 0.48 inst/cycle | -1.93 | 2.29 inst/cycle | -0.12 | 2.50 inst/cycle | +0.09 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_larger_block_n_and_single_pass_row | 0.946× | 15.026× | ✓ |  |
| B | increase_tile_sizes_and_reduce_register_pressure_via_autotune_xblock | 0.973× | 1.430× | ✓ |  |
| C | rebalance_autotune_configs_reduce_num_warps_and_fix_grid_to_fill_sms | 0.896× | 1.411× | ✓ |  |
| D | reduce_register_pressure_by_simplifying_in_register_reversal | 0.836× | 0.814× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | increase_tile_siz... | Δ | rebalance_autotun... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 40420 ns | 2690 ns | -37730 ns (-93.3%) | 28260 ns | -12160 ns (-30.1%) | 28640 ns | -11780 ns (-29.1%) | 49630 ns | +9210 ns (+22.8%) |
| SM Throughput (%) | 1.79 % | 0.18 % | -1.61 pp | 0.90 % | -0.89 pp | 0.89 % | -0.90 pp | 1.06 % | -0.73 pp |
| Memory Throughput (%) | 1.50 % | 0.39 % | -1.11 pp | 1.05 % | -0.45 pp | 1.04 % | -0.46 pp | 1.59 % | +0.09 pp |
| DRAM Throughput (%) | 0.04 % | 0.05 % | +0.01 pp | 0.07 % | +0.03 pp | 0.07 % | +0.03 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 49.92 % | 6.24 % | -43.68 pp | 12.49 % | -37.43 pp | 12.49 % | -37.43 pp | 24.98 % | -24.94 pp |
| Registers/Thread | 64.00 register/thread | 47.00 register/thread | -17 | 255.00 register/thread | +191 | 255.00 register/thread | +191 | 64.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.00 | -0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.02 | -0.01 |
| No Eligible (%) | 39.50 % | 86.25 % | +46.75 pp | 71.29 % | +31.79 pp | 70.55 % | +31.05 pp | 65.90 % | +26.40 pp |
| IPC Active | 2.41 inst/cycle | 0.52 inst/cycle | -1.89 | 1.15 inst/cycle | -1.26 | 1.18 inst/cycle | -1.23 | 1.37 inst/cycle | -1.04 |


### Key Learnings

#### ❌ Plan D: `reduce_register_pressure_by_simplifying_in_register_reversal` (0.814×)

Switching autotune configs to favor very large BLOCK_N values (32768, 16384, 8192) with large num_warps caused the autotuner to select a configuration with 512 threads/block and 64 registers/thread, raising Block Limit Registers from 10 to 2 and increasing kernel duration from 2.82µs to 49.63µs (17.6x slower). The large block sizes forced the compiler to spill registers to local memory, triggering CPIStall warnings (30.2% of cycles stalled on L1TEX) and MemoryCacheAccessPattern warnings for uncoalesced local loads/stores. The structural anti-pattern is: adding large-BLOCK_N autotune configs to a kernel that is already register-pressure-limited causes the autotuner to select a config that maximizes register spill, not throughput.

> **Applicable when**: A Triton kernel regresses after adding large-BLOCK_N autotune configs; registers/thread increases and Block Limit Registers drops to 1-2; CPIStall and MemoryCacheAccessPattern rules fire for local loads/stores (register spill traffic); kernel duration increases by >5x.

> **Avoid when**: The kernel is memory-bandwidth-bound (DRAM throughput >50%) and large tiles are needed to amortize memory latency; or the kernel has low register usage (<32 regs/thread) where large BLOCK_N does not cause spill.

---

## Cumulative Progress

```
Baseline:  40420 ns  (1.000×)

Final best: 40420 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,13 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_N': 32768, 'XBLOCK': 1}, num_warps=32, num_stages=1),
-        triton.Config({'BLOCK_N': 16384, 'XBLOCK': 1}, num_warps=16, num_stages=1),
-        triton.Config({'BLOCK_N': 8192, 'XBLOCK': 1}, num_warps=8, num_stages=1),
-        triton.Config({'BLOCK_N': 4096, 'XBLOCK': 1}, num_warps=4, num_stages=1),
-        triton.Config({'BLOCK_N': 2048, 'XBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_N': 512,  'XBLOCK': 1}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_N': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_N': 2048, 'XBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_N': 4096, 'XBLOCK': 1}, num_warps=8, num_stages=2),
     ],
     key=['N'],
 )
@@ -25,78 +24,41 @@

 ):
     # Each program handles one row
     row_id = tl.program_id(0)
-    
+
     if row_id >= M:
         return
-    
+
     row_base_in = in_ptr + row_id * N
     row_base_out = out_ptr + row_id * N
-    
-    # We need to compute reverse cumsum:
-    # out[i] = sum(x[i:])
-    # This is equivalent to: flip -> cumsum -> flip
-    # 
-    # Strategy: iterate from right to left, accumulating sum
-    # We process in blocks of BLOCK_N, going from the end to the start
-    
-    # We'll do a two-pass approach:
-    # Pass 1: compute block sums from right to left
-    # Pass 2: compute prefix sums within each block, offset by block suffix sums
-    
-    # For large N, we need to handle multiple blocks
-    # We iterate from right to left
-    
-    # Number of blocks
+
     num_blocks = (N + BLOCK_N - 1) // BLOCK_N
-    
-    # We'll accumulate the running sum from the right
+
     running_sum = tl.zeros([1], dtype=tl.float32)
-    
+
     # Process blocks from right to left
     for block_idx in range(num_blocks - 1, -1, -1):
         block_start = block_idx * BLOCK_N
-        offsets = block_start + tl.arange(0, BLOCK_N)
-        mask = offsets < N
-        
-        # Load values
-        vals = tl.load(row_base_in + offsets, mask=mask, other=0.0).to(tl.float32)
-        
-        # Compute suffix sum within this block (from right to left)
-        # We need: out[j] = sum(vals[j:block_end]) + running_sum
-        # This is: reverse cumsum of vals + running_sum
-        
-        # Flip the block, compute cumsum, flip back
-        # Since BLOCK_N is constexpr, we can use associative scan
-        # We need to reverse the indices within the block
-        
-        # Reverse the values within the block
+
+        # Single reversed load — eliminates the redundant forward 'vals' load
         rev_offsets = (BLOCK_N - 1) - tl.arange(0, BLOCK_N)
-        # Map reversed offsets to actual positions
         actual_rev_offsets = block_start + rev_offsets
         rev_mask = actual_rev_offsets < N
         rev_vals = tl.load(row_base_in + actual_rev_offsets, mask=rev_mask, other=0.0).to(tl.float32)
-        
+
         # Compute prefix cumsum on reversed values
-        # Use associative scan
         rev_cumsum = tl.cumsum(rev_vals, axis=0)
-        
+
         # Add running_sum to all elements
         rev_cumsum = rev_cumsum + running_sum
-        
-        # Update running_sum: total sum of this block
-        block_total = tl.sum(vals, axis=0)
+
+        # Update running_sum: total sum of this block (reuse rev_vals, no separate forward load)
+        block_total = tl.sum(rev_vals, axis=0)
         running_sum = running_sum + block_total
-        
-        # Now we need to store: the reversed cumsum back in original order
-        # rev_cumsum[k] corresponds to position block_start + (BLOCK_N-1-k)
-        # So out[block_start + j] = rev_cumsum[BLOCK_N-1-j]
-        # We store by reversing the output indices
-        
+
         # Store back: position block_start + (BLOCK_N-1-k) gets rev_cumsum[k]
-        store_offsets = block_start + rev_offsets  # = block_start + (BLOCK_N-1) - arange(BLOCK_N)
+        store_offsets = block_start + rev_offsets
         store_mask = store_offsets < N
         tl.store(row_base_out + store_offsets, rev_cumsum.to(tl.float32), mask=store_mask)
-
 
 def kernel_function(x):
     assert x.is_cuda and x.dtype == torch.float32
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
