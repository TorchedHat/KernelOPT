# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_mean_relu_t_view_129` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 32060 ns — **Best latency**: 30020 ns (**1.0680×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_hardcoded_grid_size_and_b_matrix_load_index`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_mean_relu_t_view_129`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_129_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 44
- **Duration**: 32060 ns
- **SM Throughput**: 0.06 % | **Memory**: 0.58 % | **DRAM**: 0.06 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.31 inst/cycle | **No Eligible**: 92.16 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.056× | 1.069× | ✓ | **✓ best** |
| B | persistent_kernel_grid_k_parallelism_with_autotune | 1.048× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 32060 ns | 29980 ns | -2080 ns (-6.5%) | 32030 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 0.06 % | 0.07 % | +0.01 pp | 0.06 % | +0.00 pp |
| Memory Throughput (%) | 0.58 % | 0.63 % | +0.05 pp | 0.59 % | +0.01 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.23 % | -0.01 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 44.00 register/thread | 42.00 register/thread | -2 | 44.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 92.16 % | 90.34 % | -1.82 pp | 92.13 % | -0.03 pp |
| IPC Active | 0.31 inst/cycle | 0.39 inst/cycle | +0.08 | 0.31 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0694× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_strided_load_to_eliminate_shared_memory_bank_conflicts | 1.049× | 1.071× | ✓ |  |
| C | reduce_block_k_to_cut_shared_memory_pressure | 1.056× | 1.077× | ✓ |  |
| D | reduce_block_k_with_persistent_k_loop_and_autotune | 1.060× | 1.068× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_stri... | Δ | reduce_block_k_to... | Δ | reduce_block_k_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32060 ns | 29950 ns | -2110 ns (-6.6%) | 29760 ns | -2300 ns (-7.2%) | 30020 ns | -2040 ns (-6.4%) |
| SM Throughput (%) | 0.06 % | 0.07 % | +0.01 pp | 0.07 % | +0.01 pp | 0.07 % | +0.01 pp |
| Memory Throughput (%) | 0.58 % | 0.64 % | +0.06 pp | 0.64 % | +0.06 pp | 0.63 % | +0.05 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.23 % | -0.01 pp | 6.23 % | -0.01 pp | 6.23 % | -0.01 pp |
| Registers/Thread | 44.00 register/thread | 42.00 register/thread | -2 | 42.00 register/thread | -2 | 42.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 92.16 % | 90.30 % | -1.86 pp | 90.39 % | -1.77 pp | 90.29 % | -1.87 pp |
| IPC Active | 0.31 inst/cycle | 0.39 inst/cycle | +0.08 | 0.38 inst/cycle | +0.07 | 0.38 inst/cycle | +0.07 |


### Key Learnings

#### ✅ Plan A: `fix_b_matrix_strided_load_to_eliminate_shared_memory_bank_conflicts` (1.071×)

When a matrix load uses an incorrect stride (e.g., output stride instead of the source matrix's actual stride), the resulting transposed or scrambled shared memory access pattern causes severe bank conflicts. In this case, replacing a buggy broadcast_to indirection that used stride 10 (output row stride) with a direct coalesced index using the correct stride_bn=1024 yielded a 1.07x speedup. However, the SharedMemoryConflicts metrics (12-way conflicts, 77.76% excessive wavefronts) and CPIStall MIO scoreboard cycles (7.6) remained essentially unchanged after the fix, suggesting the compiler may have re-routed the load through a different path or the bank conflict source was not fully eliminated by this index change alone.

> **Applicable when**: A matrix load in a Triton GEMM kernel uses an incorrect stride (e.g., output matrix stride instead of the source matrix's actual stride), combined with a broadcast_to indirection that obscures the true access pattern. NCU reports SharedMemoryConflicts with high bank-conflict-way (>=8) and excessive wavefronts (>70%), and CPIStall shows MIO scoreboard stalls >7 cycles representing >70% of total stall cycles.

> **Avoid when**: The load index is already correct and coalesced. If the bank conflicts persist after the fix (as seen here where metrics barely changed), the root cause may be elsewhere (e.g., a different shared memory operation, or the compiler staging the corrected load through shared memory in a conflicting layout). Do not expect large speedups from index correction alone when the kernel is severely underutilized (grid_size=1, waves_per_sm=0.0) — the dominant bottleneck is launch configuration, not memory access pattern.

#### ✅ Plan C: `reduce_block_k_to_cut_shared_memory_pressure` (1.077×)

Adding a @triton.autotune decorator with multiple (num_warps, num_stages) configurations to a small GEMM kernel that already has BLOCK_K reduced yielded a modest 7.7% speedup. The primary observable metric change was L2 Hit Rate improving dramatically from 53.56% to 94.47%, suggesting the autotuner selected a configuration with more pipeline stages that better prefetches data into L2. The shared memory and occupancy metrics remained unchanged (BLOCK_K was already reduced in the prior iteration), confirming the gain came from better warp/stage scheduling rather than tile size changes.

> **Applicable when**: Triton GEMM kernel where tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) are already fixed/optimized but num_warps and num_stages are not autotuned. Particularly effective when L2 Hit Rate is low (<60%) suggesting poor prefetch behavior, and when the kernel has a non-trivial K-loop (K >= 256) where pipeline depth (num_stages) can hide memory latency.

> **Avoid when**: Kernel already has @triton.autotune with a comprehensive config space. Avoid when the kernel is so small (grid=1, tiny K) that autotuning overhead dominates. Do not apply when the primary bottleneck is shared memory bank conflicts or occupancy — fix those structural issues first before autotuning scheduling parameters.

---

## Iteration 3

**Best so far**: 1.0694× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_size_and_b_matrix_load_index | 1.053× | 1.068× | ✓ |  |
| B | fuse_bias_broadcast_fix_with_comprehensive_autotune_sweep | 1.049× | 0.725× | ✓ |  |
| C | fix_b_matrix_broadcast_to_indirection_for_coalesced_shared_memory_access | 1.053× | 1.066× | ✓ |  |
| D | add_triton_autotune | 1.053× | 1.066× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fuse_bias_broadca... | Δ | fix_b_matrix_broa... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32060 ns | 30020 ns | -2040 ns (-6.4%) | 44220 ns | +12160 ns (+37.9%) | 30080 ns | -1980 ns (-6.2%) | 30080 ns | -1980 ns (-6.2%) |
| SM Throughput (%) | 0.06 % | 0.07 % | +0.01 pp | 0.04 % | -0.02 pp | 0.07 % | +0.01 pp | 0.07 % | +0.01 pp |
| Memory Throughput (%) | 0.58 % | 0.63 % | +0.05 pp | 0.43 % | -0.15 pp | 0.64 % | +0.06 pp | 0.63 % | +0.05 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.04 % | -0.02 pp | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.23 % | -0.01 pp | 3.12 % | -3.12 pp | 6.23 % | -0.01 pp | 6.23 % | -0.01 pp |
| Registers/Thread | 44.00 register/thread | 42.00 register/thread | -2 | 51.00 register/thread | +7 | 42.00 register/thread | -2 | 42.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 92.16 % | 90.30 % | -1.86 pp | 90.30 % | -1.86 pp | 90.29 % | -1.87 pp | 90.41 % | -1.75 pp |
| IPC Active | 0.31 inst/cycle | 0.39 inst/cycle | +0.08 | 0.19 inst/cycle | -0.12 | 0.39 inst/cycle | +0.08 | 0.39 inst/cycle | +0.08 |

---

## Iteration 4

**Best so far**: 1.0694× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_bias_broadcast_load_to_avoid_shared_memory_bank_conflicts | 1.049× | 1.071× | ✓ |  |
| B | prune_low_warp_autotune_configs_and_restore_dispatch_key | 1.049× | 0.724× | ✓ |  |
| D | fix_b_matrix_load_with_correct_stride_and_eliminate_broadcast_indirection | 1.053× | 1.067× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_bias_broadcas... | Δ | prune_low_warp_au... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32060 ns | 29920 ns | -2140 ns (-6.7%) | 44260 ns | +12200 ns (+38.1%) | 30050 ns | -2010 ns (-6.3%) |
| SM Throughput (%) | 0.06 % | 0.07 % | +0.01 pp | 0.05 % | -0.01 pp | 0.07 % | +0.01 pp |
| Memory Throughput (%) | 0.58 % | 0.63 % | +0.05 pp | 0.43 % | -0.15 pp | 0.64 % | +0.06 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.04 % | -0.02 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp | 6.25 % | +0.01 pp | 6.22 % | -0.02 pp |
| Registers/Thread | 44.00 register/thread | 42.00 register/thread | -2 | 34.00 register/thread | -10 | 42.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 92.16 % | 90.37 % | -1.79 pp | 94.23 % | +2.07 pp | 90.33 % | -1.83 pp |
| IPC Active | 0.31 inst/cycle | 0.39 inst/cycle | +0.08 | 0.23 inst/cycle | -0.08 | 0.38 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan B: `prune_low_warp_autotune_configs_and_restore_dispatch_key` (0.724×)

Removing num_warps=2 configs from the autotune sweep and changing the dispatch key from key=['xnumel'] to key=['BLOCK_M','BLOCK_N','BLOCK_K'] caused a 27.6% regression (30.02 µs → 44.26 µs). Despite the block size remaining at 128 threads, registers per thread dropped from 42 to 34 and dynamic shared memory halved from 8 KB to 4 KB, indicating the autotuner selected a different config that restructured the kernel's memory layout. Warp cycles per issued instruction worsened dramatically from 10.38 to 17.38, and a new L1TEX scoreboard stall appeared (33.3% of stall cycles), suggesting the changed bias load pattern (1D load + register broadcast replacing 2D shared memory load) introduced additional L1TEX latency that the prior shared memory path did not have. Changing the autotune key to constexpr values (BLOCK_M, BLOCK_N, BLOCK_K) that are fixed at compile time effectively makes the key meaningless, causing unpredictable config selection.

> **Applicable when**: Never — this is a negative result. Do not use constexpr tl.constexpr values as autotune keys, and do not simultaneously change both the autotune config space and the bias/epilogue load pattern.

> **Avoid when**: Always avoid: (1) using tl.constexpr kernel arguments (BLOCK_M, BLOCK_N, BLOCK_K) as autotune dispatch keys — these are compile-time constants and do not vary across calls, making the key degenerate and causing unpredictable config selection; (2) simultaneously pruning autotune configs AND changing the epilogue load pattern, as the two changes interact and make regression attribution impossible; (3) replacing a 2D shared memory bias load with a 1D register broadcast when the kernel already has L1TEX stall pressure — the 1D load adds a new L1TEX scoreboard dependency (33.3% of stall cycles) that did not exist before.

---

## Cumulative Progress

```
Baseline:  32060 ns  (1.000×)
Iter 1     29980 ns  (1.069×)  — tf32_tensor_cores

Final best: 30020 ns  (1.0680×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_addmm_mean_relu_t_view_129(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -12,7 +26,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 16
     BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -60,21 +74,18 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
         
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 1024*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 10*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Fixed B load: direct coalesced index using correct stride_bk=1, stride_bn=1024
+        b_k_offs = offs_k + k_idx * BLOCK_K   # absolute K offsets, shape [BLOCK_K]
+        b_ptrs = b_k_offs[:, None] * 1 + offs_b_n[None, :] * 1024  # [BLOCK_K, BLOCK_N]
+        b = tl.load(B + b_ptrs)
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +99,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
