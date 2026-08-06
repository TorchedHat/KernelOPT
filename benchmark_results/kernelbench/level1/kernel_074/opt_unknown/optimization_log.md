# Kernel Optimization Log

## Summary

- **Kernel**: `conv_transpose1d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 28550000 ns — **Best latency**: 16770000 ns (**1.7024×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_static_range_with_range_to_eliminate_register_explosion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv_transpose1d_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_074/synthesis/kernel_0_unknown.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2099200,1,1), Waves/SM: 993.94
- **Registers/thread**: 20
- **Duration**: 28550000 ns
- **SM Throughput**: 87.97 % | **Memory**: 51.36 % | **DRAM**: 1.13 %
- **Achieved Occupancy**: 93.29 %
- **IPC Active**: 3.52 inst/cycle | **No Eligible**: 12.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_weight_loads_and_vectorize_inner_loop | 0.628× | 0.766× | ✓ |  |
| B | eliminate_redundant_scalar_loads_and_fix_inner_loop | — | fail | ✓ |  |
| C | eliminate_redundant_scalar_weight_loads_and_fix_vectorized_accumulation | 1.001× | 1.174× | ✓ | **✓ best** |
| D | eliminate_redundant_weight_loads_and_vectorize_inner_accumulation | 1.003× | 1.174× | ✓ |  |
| E | add_triton_autotune | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | eliminate_redunda... | Δ | eliminate_redunda... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 28550000 ns | 37260000 ns | +8710000 ns (+30.5%) | 24320000 ns | -4230000 ns (-14.8%) | 24320000 ns | -4230000 ns (-14.8%) |
| SM Throughput (%) | 87.97 % | 79.90 % | -8.07 pp | 76.34 % | -11.63 pp | 76.34 % | -11.63 pp |
| Memory Throughput (%) | 51.36 % | 93.43 % | +42.07 pp | 60.28 % | +8.92 pp | 60.28 % | +8.92 pp |
| DRAM Throughput (%) | 1.13 % | 0.87 % | -0.26 pp | 1.33 % | +0.20 pp | 1.33 % | +0.20 pp |
| Achieved Occupancy (%) | 93.29 % | 73.98 % | -19.31 pp | 61.06 % | -32.23 pp | 61.13 % | -32.16 pp |
| Registers/Thread | 20.00 register/thread | 40.00 register/thread | +20 | 42.00 register/thread | +22 | 42.00 register/thread | +22 |
| Waves/SM | 993.94 | 1325.25 | +331.31 | 1590.30 | +596.36 | 1590.30 | +596.36 |
| No Eligible (%) | 12.01 % | 20.09 % | +8.08 pp | 23.64 % | +11.63 pp | 23.66 % | +11.65 pp |
| IPC Active | 3.52 inst/cycle | 3.20 inst/cycle | -0.32 | 3.05 inst/cycle | -0.47 | 3.05 inst/cycle | -0.47 |

---

## Iteration 2

**Best so far**: 1.1739× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_smaller_block_ic_configs | 1.037× | 1.642× | ✓ | **✓ best** |
| C | replace_one_hot_scalar_extraction_with_2d_tile_load_and_static_range | 1.005× | 1.702× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | replace_one_hot_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 28550000 ns | 17390000 ns | -11160000 ns (-39.1%) | 16770000 ns | -11780000 ns (-41.3%) |
| SM Throughput (%) | 87.97 % | 70.90 % | -17.07 pp | 72.36 % | -15.61 pp |
| Memory Throughput (%) | 51.36 % | 68.38 % | +17.02 pp | 64.30 % | +12.94 pp |
| DRAM Throughput (%) | 1.13 % | 1.86 % | +0.73 pp | 1.93 % | +0.80 pp |
| Achieved Occupancy (%) | 93.29 % | 70.40 % | -22.89 pp | 73.89 % | -19.40 pp |
| Registers/Thread | 20.00 register/thread | 40.00 register/thread | +20 | 34.00 register/thread | +14 |
| Waves/SM | 993.94 | 663.27 | -330.67 | 663.27 | -330.67 |
| No Eligible (%) | 12.01 % | 29.07 % | +17.06 pp | 27.61 % | +15.60 pp |
| IPC Active | 3.52 inst/cycle | 2.84 inst/cycle | -0.68 | 2.90 inst/cycle | -0.62 |

---

## Iteration 3

**Best so far**: 1.6417× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_block_l_and_num_stages_to_hide_strided_ic_loads | 1.021× | 1.702× | ✓ | **✓ best** |
| C | hoist_kernel_loop_and_fuse_fma_via_loop_reorder | — | fail | ✓ |  |
| D | replace_elementwise_multiply_reduce_with_tl_dot_fma | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_l_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 28550000 ns | 16770000 ns | -11780000 ns (-41.3%) |
| SM Throughput (%) | 87.97 % | 72.36 % | -15.61 pp |
| Memory Throughput (%) | 51.36 % | 64.31 % | +12.95 pp |
| DRAM Throughput (%) | 1.13 % | 1.93 % | +0.80 pp |
| Achieved Occupancy (%) | 93.29 % | 73.89 % | -19.40 pp |
| Registers/Thread | 20.00 register/thread | 34.00 register/thread | +14 |
| Waves/SM | 993.94 | 663.27 | -330.67 |
| No Eligible (%) | 12.01 % | 27.61 % | +15.60 pp |
| IPC Active | 3.52 inst/cycle | 2.90 inst/cycle | -0.62 |

---

## Iteration 4

**Best so far**: 1.7024× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_loop_order_k_outer_ic_inner_to_hoist_index_arithmetic_and_enable_fma | 0.961× | 1.295× | ✓ |  |
| B | preload_weight_slice_into_register_cache_to_eliminate_strided_global_loads | 0.539× | 0.287× | ✓ |  |
| C | hoist_x_tile_load_outside_k_loop_to_amortize_strided_dram_access | 0.570× | 0.287× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_loop_order_k... | Δ | preload_weight_sl... | Δ | hoist_x_tile_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 28550000 ns | 22040000 ns | -6510000 ns (-22.8%) | 99500000 ns | +70950000 ns (+248.5%) | 99320000 ns | +70770000 ns (+247.9%) |
| SM Throughput (%) | 87.97 % | 82.21 % | -5.76 pp | 26.07 % | -61.90 pp | 26.12 % | -61.85 pp |
| Memory Throughput (%) | 51.36 % | 54.73 % | +3.37 pp | 96.86 % | +45.50 pp | 97.09 % | +45.73 pp |
| DRAM Throughput (%) | 1.13 % | 1.47 % | +0.34 pp | 0.33 % | -0.80 pp | 0.33 % | -0.80 pp |
| Achieved Occupancy (%) | 93.29 % | 98.30 % | +5.01 pp | 31.17 % | -62.12 pp | 31.17 % | -62.12 pp |
| Registers/Thread | 20.00 register/thread | 30.00 register/thread | +10 | 96.00 register/thread | +76 | 96.00 register/thread | +76 |
| Waves/SM | 993.94 | 497.45 | -496.49 | 1591.85 | +597.91 | 1591.85 | +597.91 |
| No Eligible (%) | 12.01 % | 17.77 % | +5.76 pp | 79.42 % | +67.41 pp | 79.52 % | +67.51 pp |
| IPC Active | 3.52 inst/cycle | 3.29 inst/cycle | -0.23 | 0.82 inst/cycle | -2.70 | 0.82 inst/cycle | -2.70 |


### Key Learnings

#### ❌ Plan B: `preload_weight_slice_into_register_cache_to_eliminate_strided_global_loads` (0.287×)

Hoisting KERNEL_SIZE separate weight loads (w_col_0 … w_col_4) outside the k-loop to eliminate repeated strided global loads caused a catastrophic register explosion: registers/thread jumped from 34 to 96, collapsing theoretical occupancy from 75% to 31.25% and achieved occupancy from 73.9% to 31.2%. The Block Limit Registers fell from 12 to 5 blocks/SM, waves/SM more than doubled (663 → 1592), and eligible warps/scheduler crashed from 5.3 to 0.3 — leaving the scheduler idle 79% of cycles (No Eligible: 79.4%). The kernel became severely L1/TEX memory-bound (Mem Busy: 64% → 97%) with compute throughput dropping from 72% to 26%, and total executed instructions increased by 69% (9.6B → 16.2B) due to the additional shared memory bank conflicts (4.5-way conflicts on 77.8% of shared load/store wavefronts) introduced by the static_range unrolling. The strategy is fundamentally wrong when the register budget is already near the occupancy cliff: holding KERNEL_SIZE live register vectors across the k-loop body forces the compiler to keep all w_col_k arrays live simultaneously, consuming far more registers than the per-iteration load they replace.

> **Applicable when**: Never applicable in this form. Do NOT hoist multiple KERNEL_SIZE register vectors outside a loop when registers/thread are already ≥30 and theoretical occupancy is ≤75% — the simultaneous liveness of all hoisted vectors will multiply register consumption by KERNEL_SIZE.

> **Avoid when**: Registers/thread are already ≥30 (Block Limit Registers is already the binding occupancy constraint); KERNEL_SIZE ≥ 3 (holding KERNEL_SIZE live [BLOCK_IC] vectors simultaneously multiplies register use by KERNEL_SIZE); the kernel already has a 2D tile [BLOCK_IC, BLOCK_L] live in the loop body (compounding register pressure); tl.static_range unrolling is used (forces all hoisted vectors to remain live across the entire unrolled body, preventing register reuse). The correct fix for strided weight loads is a 2D tl.load of shape [BLOCK_IC, KERNEL_SIZE] into a single tensor (not KERNEL_SIZE separate 1D loads), but only if the register budget can absorb BLOCK_IC*KERNEL_SIZE additional registers.

---

## Iteration 5

**Best so far**: 1.7024× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.936× | 1.292× | ✓ |  |
| B | replace_static_range_with_range_to_eliminate_register_explosion | 1.043× | 1.702× | ✓ | **✓ best** |
| C | revert_hoisted_weight_loads_to_per_k_load_to_restore_occupancy | 1.043× | 1.642× | ✓ |  |
| D | add_triton_autotune | 0.963× | 1.295× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | replace_static_ra... | Δ | revert_hoisted_we... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 28550000 ns | 22100000 ns | -6450000 ns (-22.6%) | 16770000 ns | -11780000 ns (-41.3%) | 17390000 ns | -11160000 ns (-39.1%) | 22040000 ns | -6510000 ns (-22.8%) |
| SM Throughput (%) | 87.97 % | 82.46 % | -5.51 pp | 72.37 % | -15.60 pp | 70.90 % | -17.07 pp | 82.21 % | -5.76 pp |
| Memory Throughput (%) | 51.36 % | 55.00 % | +3.64 pp | 64.30 % | +12.94 pp | 68.48 % | +17.12 pp | 54.73 % | +3.37 pp |
| DRAM Throughput (%) | 1.13 % | 1.47 % | +0.34 pp | 1.93 % | +0.80 pp | 1.86 % | +0.73 pp | 1.47 % | +0.34 pp |
| Achieved Occupancy (%) | 93.29 % | 98.30 % | +5.01 pp | 73.89 % | -19.40 pp | 70.53 % | -22.76 pp | 98.28 % | +4.99 pp |
| Registers/Thread | 20.00 register/thread | 30.00 register/thread | +10 | 34.00 register/thread | +14 | 40.00 register/thread | +20 | 30.00 register/thread | +10 |
| Waves/SM | 993.94 | 497.45 | -496.49 | 663.27 | -330.67 | 663.27 | -330.67 | 497.45 | -496.49 |
| No Eligible (%) | 12.01 % | 17.52 % | +5.51 pp | 27.61 % | +15.60 pp | 29.07 % | +17.06 pp | 17.77 % | +5.76 pp |
| IPC Active | 3.52 inst/cycle | 3.30 inst/cycle | -0.22 | 2.90 inst/cycle | -0.62 | 2.84 inst/cycle | -0.68 | 3.29 inst/cycle | -0.23 |


### Key Learnings

#### ✅ Plan A: `persistent_tile_scheduling` (1.292×)

Restructuring a Triton kernel from a one-CTA-per-tile grid to a persistent while-loop (capping axis-1 grid size at num_SMs and striding over tiles) produced only a 1.29x speedup despite targeting 15% uncoalesced excessive sectors. The profiling data shows virtually no change in any NCU metric: grid size, waves/SM, registers/thread, achieved occupancy, executed instructions, L1 hit rate, L2 hit rate, and Mem Busy are all within 1% of baseline. The speedup likely comes from reduced kernel launch overhead or minor scheduling improvements rather than the intended weight-load reuse, because L2 hit rate was already 98.59% — weight data was already resident in L2 across all tiles, so the persistent loop's reuse benefit did not materialize as reduced memory traffic. When L2 hit rate is near 100% and DRAM throughput is negligible (1.47%), persistent scheduling cannot reduce memory traffic further.

> **Applicable when**: Kernel has a large grid (many waves/SM) and DRAM throughput is non-trivial (>10%), meaning weight data is NOT fully resident in L2; the persistent loop would allow each SM to reuse weight data across multiple tiles without re-fetching from DRAM; uncoalesced excessive sectors are >10% and L2 hit rate is below 90%.

> **Avoid when**: L2 hit rate is already near 100% (>98%) and DRAM throughput is negligible (<2%) — in this regime, weight data is already cached across all tiles and persistent scheduling cannot reduce memory traffic further; the kernel is compute-bound (Compute SM Throughput >80%) with no memory bottleneck to amortize; the grid axis-1 size is already small (num_tiles < num_SMs), making the persistent loop a no-op.

#### ✅ Plan C: `revert_hoisted_weight_loads_to_per_k_load_to_restore_occupancy` (1.642×)

Reverting from a k-outer/ic-inner loop order (which achieved 100% theoretical occupancy at 30 regs/thread but inflated executed instructions by 49% to 14.4B) back to an ic-outer/k-inner loop order reduced executed instructions by 32% (14.4B → 9.8B) and cut kernel duration from 22.04ms to 17.39ms (1.27x improvement for this kernel, contributing to 1.64x overall speedup). Although theoretical occupancy dropped from 100% to 75% and registers/thread rose from 30 to 40, the L1/TEX hit rate improved dramatically (60% → 83%) and warp cycles per issued instruction fell from 19.1 to 15.9, indicating that the ic-outer ordering produces better cache locality for the input tile loads. When instruction count is the dominant cost driver (not occupancy), reverting a loop reorder that inflated instructions is the correct action even at the cost of occupancy.

> **Applicable when**: A Triton kernel has been reordered to put a small-trip-count loop (e.g. KERNEL_SIZE=5) outermost to hoist index arithmetic, but NCU shows executed instructions increased by >30% after the reorder; L1/TEX hit rate is lower in the reordered version (e.g. 60% vs 83%); kernel duration increased despite higher achieved occupancy; the inner loop accesses a 2D tile [BLOCK_IC, BLOCK_L] whose cache reuse depends on the ic dimension being outermost.

> **Avoid when**: The k-outer ordering genuinely reduces total instructions (verify via Instruction Statistics.Executed Instructions); the L1/TEX hit rate is equal or better with k-outer; occupancy is the binding bottleneck (scheduler No Eligible >60% and eligible warps/scheduler <2.0) and instruction count is not inflated.

---

## Cumulative Progress

```
Baseline:  28550000 ns  (1.000×)
Iter 1     24320000 ns  (1.174×)  — eliminate_redundant_scalar_weight_loads_and_fix_vectorized_accumulation
Iter 2     17390000 ns  (1.642×)  — reduce_register_pressure_via_smaller_block_ic_configs
Iter 3     16770000 ns  (1.702×)  — increase_block_l_and_num_stages_to_hide_strided_ic_loads

Final best: 16770000 ns  (1.7024×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,32 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_L': 128, 'BLOCK_IC': 8}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_L': 256, 'BLOCK_IC': 8}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_L': 512, 'BLOCK_IC': 4}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_L': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_L': 256, 'BLOCK_IC': 16}, num_warps=8, num_stages=3),
-        triton.Config({'BLOCK_L': 128, 'BLOCK_IC': 16}, num_warps=4, num_stages=3),
+        # Existing small configs (kept for small OUT_L)
+        triton.Config({'BLOCK_L': 256, 'BLOCK_IC': 4},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_L': 512, 'BLOCK_IC': 4},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_L': 128, 'BLOCK_IC': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_L': 256, 'BLOCK_IC': 8},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_L':  64, 'BLOCK_IC': 16}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_L': 128, 'BLOCK_IC': 16}, num_warps=8,  num_stages=2),
+        # NEW: larger BLOCK_L to reduce grid size and improve w_vals reuse
+        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 4},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 4},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_L': 2048, 'BLOCK_IC': 4},  num_warps=16, num_stages=3),
+        triton.Config({'BLOCK_L': 2048, 'BLOCK_IC': 4},  num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 8},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 8},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_L': 2048, 'BLOCK_IC': 8},  num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 4},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 8},  num_warps=8,  num_stages=4),
+        # NEW: BLOCK_IC=2 configs for lower register pressure
+        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 2}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_L': 1024, 'BLOCK_IC': 2}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_L': 2048, 'BLOCK_IC': 2}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_L': 256,  'BLOCK_IC': 2}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_L': 512,  'BLOCK_IC': 2}, num_warps=4, num_stages=4),
     ],
     key=['IN_C', 'OUT_L', 'KERNEL_SIZE'],
 )
@@ -39,7 +57,11 @@

 
     # Loop over in_channels in blocks
     for ic_start in range(0, IN_C, BLOCK_IC):
-        # Loop over kernel positions
+        # Hoist ic_offs and ic_mask outside the k loop — they are k-independent
+        ic_offs = ic_start + tl.arange(0, BLOCK_IC)
+        ic_mask = ic_offs < IN_C
+
+        # Use range() NOT tl.static_range — avoids register explosion from forced unrolling
         for k in range(KERNEL_SIZE):
             # For output position o, kernel position k:
             # input position i = (o - k*dilation + padding) / stride
@@ -53,32 +75,24 @@

             valid_i = (i_idx >= 0) & (i_idx < IN_L)
             valid = out_mask & valid_stride & valid_i
 
-            # Load weights for this kernel position, for BLOCK_IC channels
-            ic_offs = ic_start + tl.arange(0, BLOCK_IC)
-            ic_mask = ic_offs < IN_C
-
             # Weight shape: (IN_C, OUT_C, KERNEL_SIZE)
-            # weight[ic, oc, k]
+            # weight[ic, oc, k] — single coalesced load for all BLOCK_IC channels
             w_ptrs = W_ptr + ic_offs * (OUT_C * KERNEL_SIZE) + oc * KERNEL_SIZE + k
             w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)  # [BLOCK_IC]
 
-            # Load input for each valid output position
+            # Single 2D tile load: [BLOCK_IC, BLOCK_L]
             # X shape: (BATCH, IN_C, IN_L)
-            # x[batch, ic, i_idx]
-            # We need to accumulate over BLOCK_IC channels
-            for ic_local in range(BLOCK_IC):
-                ic = ic_start + ic_local
-                if ic < IN_C:
-                    w_val = tl.load(W_ptr + ic * (OUT_C * KERNEL_SIZE) + oc * KERNEL_SIZE + k)
-                    x_ptrs = X_ptr + batch_idx * (IN_C * IN_L) + ic * IN_L + i_idx
-                    x_vals = tl.load(x_ptrs, mask=valid, other=0.0)  # [BLOCK_L]
-                    acc += x_vals * w_val
+            x_ptrs_2d = X_ptr + batch_idx * (IN_C * IN_L) + ic_offs[:, None] * IN_L + i_idx[None, :]
+            x_mask_2d = ic_mask[:, None] & valid[None, :]       # [BLOCK_IC, BLOCK_L]
+            x_tile = tl.load(x_ptrs_2d, mask=x_mask_2d, other=0.0)  # [BLOCK_IC, BLOCK_L]
+
+            # Vectorized accumulate — no inner loop, no one-hot, no ic_lane
+            acc += tl.sum(w_vals[:, None] * x_tile, axis=0)    # [BLOCK_L]
 
     # Store output
     # Y shape: (BATCH, OUT_C, OUT_L)
     y_ptrs = Y_ptr + batch_idx * (OUT_C * OUT_L) + oc * OUT_L + out_offs
     tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)
-
 
 def kernel_function(weight, x):
     # weight: (IN_C, OUT_C, KERNEL_SIZE) - ConvTranspose1d weight is (in_channels, out_channels, kernel_size)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
