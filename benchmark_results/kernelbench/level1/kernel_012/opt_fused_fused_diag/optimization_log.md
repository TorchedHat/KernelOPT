# Kernel Optimization Log

## Summary

- **Kernel**: `diag_matmul_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 32830 ns — **Best latency**: 33600 ns (**0.9771×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `diag_matmul_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_012/synthesis/kernel_0_fused_fused_diag.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16384,1,1), Waves/SM: 7.76
- **Registers/thread**: 18
- **Duration**: 32830 ns
- **SM Throughput**: 13.95 % | **Memory**: 68.96 % | **DRAM**: 68.96 %
- **Achieved Occupancy**: 81.16 %
- **IPC Active**: 0.60 inst/cycle | **No Eligible**: 84.61 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_vs_reuse | 1.000× | 0.977× | ✓ | **✓ best** |
| B | vectorized_loads_with_contiguous_hints | 0.996× | 0.976× | ✓ |  |
| D | increase_block_rows_for_l2_reuse_and_reduced_launch_overhead | 0.998× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | increase_block_ro... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32830 ns | 33600 ns | +770 ns (+2.3%) | 33630 ns | +800 ns (+2.4%) | 32990 ns | +160 ns (+0.5%) |
| SM Throughput (%) | 13.95 % | 13.53 % | -0.42 pp | 13.51 % | -0.44 pp | 13.86 % | -0.09 pp |
| Memory Throughput (%) | 68.96 % | 66.89 % | -2.07 pp | 67.20 % | -1.76 pp | 68.51 % | -0.45 pp |
| DRAM Throughput (%) | 68.96 % | 66.89 % | -2.07 pp | 67.20 % | -1.76 pp | 68.51 % | -0.45 pp |
| Achieved Occupancy (%) | 81.16 % | 80.58 % | -0.58 pp | 81.08 % | -0.08 pp | 81.17 % | +0.01 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 7.76 | +0.00 |
| No Eligible (%) | 84.61 % | 84.87 % | +0.26 pp | 84.81 % | +0.20 pp | 84.55 % | -0.06 pp |
| IPC Active | 0.60 inst/cycle | 0.59 inst/cycle | -0.01 | 0.59 inst/cycle | -0.01 | 0.61 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_store_evict_first_to_free_l2_for_input_reuse | 1.000× | 0.966× | ✓ |  |
| B | shared_memory_cache_diagonal_and_pipeline_b_loads | 1.000× | 0.965× | ✓ |  |
| C | add_num_stages_to_autotune_configs_for_latency_hiding | 0.999× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_store_e... | Δ | shared_memory_cac... | Δ | add_num_stages_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32830 ns | 33980 ns | +1150 ns (+3.5%) | 34020 ns | +1190 ns (+3.6%) | 32960 ns | +130 ns (+0.4%) |
| SM Throughput (%) | 13.95 % | 13.35 % | -0.60 pp | 13.60 % | -0.35 pp | 13.75 % | -0.20 pp |
| Memory Throughput (%) | 68.96 % | 66.11 % | -2.85 pp | 65.98 % | -2.98 pp | 68.72 % | -0.24 pp |
| DRAM Throughput (%) | 68.96 % | 66.11 % | -2.85 pp | 65.98 % | -2.98 pp | 68.72 % | -0.24 pp |
| Achieved Occupancy (%) | 81.16 % | 80.69 % | -0.47 pp | 80.73 % | -0.43 pp | 80.33 % | -0.83 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 7.76 | +0.00 |
| No Eligible (%) | 84.61 % | 84.97 % | +0.36 pp | 84.88 % | +0.27 pp | 84.53 % | -0.08 pp |
| IPC Active | 0.60 inst/cycle | 0.59 inst/cycle | -0.01 | 0.59 inst/cycle | -0.01 | 0.60 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_a_evict_last_b_evict_first | 0.968× | 0.986× | ✓ |  |
| B | cache_modifier_cg_for_b_load_to_bypass_l1_and_maximize_l2_reuse_of_a | 0.994× | 0.994× | ✓ |  |
| C | cache_all_reused_diagonal_vector_to_maximize_l1_hit_rate | 0.995× | 0.971× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_a... | Δ | cache_modifier_cg... | Δ | cache_all_reused_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32830 ns | 33310 ns | +480 ns (+1.5%) | 33020 ns | +190 ns (+0.6%) | 33820 ns | +990 ns (+3.0%) |
| SM Throughput (%) | 13.95 % | 17.23 % | +3.28 pp | 13.82 % | -0.13 pp | 13.97 % | +0.02 pp |
| Memory Throughput (%) | 68.96 % | 67.56 % | -1.40 pp | 67.98 % | -0.98 pp | 66.49 % | -2.47 pp |
| DRAM Throughput (%) | 68.96 % | 67.56 % | -1.40 pp | 67.98 % | -0.98 pp | 66.49 % | -2.47 pp |
| Achieved Occupancy (%) | 81.16 % | 81.42 % | +0.26 pp | 80.46 % | -0.70 pp | 80.70 % | -0.46 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 7.76 | +0.00 |
| No Eligible (%) | 84.61 % | 80.11 % | -4.50 pp | 84.72 % | +0.11 pp | 84.67 % | +0.06 pp |
| IPC Active | 0.60 inst/cycle | 0.77 inst/cycle | +0.17 | 0.60 inst/cycle | +0.00 | 0.59 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  32830 ns  (1.000×)

Final best: 33600 ns  (0.9771×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,25 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 1024}, num_warps=4),
-        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 512}, num_warps=4),
-        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 512}, num_warps=4),
-        triton.Config({'BLOCK_ROWS': 4, 'BLOCK_COLS': 256}, num_warps=4),
-        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 256}, num_warps=2),
-        triton.Config({'BLOCK_ROWS': 8, 'BLOCK_COLS': 128}, num_warps=4),
-        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 2048}, num_warps=8),
-        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 1024}, num_warps=8),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 4, 'BLOCK_COLS': 256},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 256},  num_warps=2, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 8, 'BLOCK_COLS': 128},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 1024}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 512},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 512},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 4, 'BLOCK_COLS': 256},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 256},  num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 8, 'BLOCK_COLS': 128},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 2048}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 1024}, num_warps=8, num_stages=3),
     ],
     key=['num_rows', 'num_cols'],
 )
@@ -37,8 +46,14 @@

     row_start = pid_row * BLOCK_ROWS
     col_start = pid_col * BLOCK_COLS
 
-    row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
-    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
+    row_offsets = tl.max_contiguous(
+        tl.multiple_of(row_start + tl.arange(0, BLOCK_ROWS), BLOCK_ROWS),
+        BLOCK_ROWS
+    )
+    col_offsets = tl.max_contiguous(
+        tl.multiple_of(col_start + tl.arange(0, BLOCK_COLS), BLOCK_COLS),
+        BLOCK_COLS
+    )
 
     row_mask = row_offsets < num_rows
     col_mask = col_offsets < num_cols
@@ -57,7 +72,6 @@

     # Store result
     out_ptrs = out_ptr + row_offsets[:, None] * stride_out_row + col_offsets[None, :] * stride_out_col
     tl.store(out_ptrs, out_vals, mask=full_mask)
-
 
 def kernel_function(a, b):
     assert a.ndim == 1
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
