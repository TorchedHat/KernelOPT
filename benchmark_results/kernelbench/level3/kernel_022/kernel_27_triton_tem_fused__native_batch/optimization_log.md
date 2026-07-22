# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 48670 ns — **Best latency**: 48700 ns (**0.9994×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_27_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.19
- **Registers/thread**: 94
- **Duration**: 48670 ns
- **SM Throughput**: 12.50 % | **Memory**: 85.19 % | **DRAM**: 1.65 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 1.001× | ✓ |  |
| B | add_autotune_to_fix_bank_conflicts_and_occupancy | 1.427× | 0.869× | ✓ |  |
| C | add_eviction_policies_and_reduce_register_pressure | 1.004× | 1.000× | ✓ |  |
| D | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 1.431× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_to_f... | Δ | add_eviction_poli... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48670 ns | 48610 ns | -60 ns (-0.1%) | 56000 ns | +7330 ns (+15.1%) | 48670 ns | +0 ns (+0.0%) | 48700 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 12.50 % | 12.48 % | -0.02 pp | 10.50 % | -2.00 pp | 12.49 % | -0.01 pp | 12.48 % | -0.02 pp |
| Memory Throughput (%) | 85.19 % | 85.07 % | -0.12 pp | 73.35 % | -11.84 pp | 85.11 % | -0.08 pp | 85.05 % | -0.14 pp |
| DRAM Throughput (%) | 1.65 % | 1.65 % | +0.00 pp | 1.43 % | -0.22 pp | 1.65 % | +0.00 pp | 1.65 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 80.00 register/thread | -14 | 94.00 register/thread | +0 | 94.00 register/thread | +0 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.16 | -0.03 | 0.19 | +0.00 | 0.19 | +0.00 |
| No Eligible (%) | 86.11 % | 86.12 % | +0.01 pp | 88.29 % | +2.18 pp | 86.06 % | -0.05 pp | 86.10 % | -0.01 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.47 inst/cycle | -0.08 | 0.56 inst/cycle | +0.01 | 0.56 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | autotune_tile_sizes_as_constexpr_to_break_bank_conflicts | 0.949× | 0.843× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 48670 ns | 57730 ns | +9060 ns (+18.6%) |
| SM Throughput (%) | 12.50 % | 19.82 % | +7.32 pp |
| Memory Throughput (%) | 85.19 % | 76.29 % | -8.90 pp |
| DRAM Throughput (%) | 1.65 % | 1.47 % | -0.18 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 |
| Waves/SM | 0.19 | 0.23 | +0.04 |
| No Eligible (%) | 86.11 % | 77.93 % | -8.18 pp |
| IPC Active | 0.55 inst/cycle | 0.88 inst/cycle | +0.33 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | 0.947× | 0.846× | ✓ |  |
| B | tma_loads | 0.938× | 0.825× | ✓ |  |
| C | add_eviction_policies_streaming_A_reused_B | 0.945× | 0.840× | ✓ |  |
| D | fix_b_matrix_load_index_to_enable_correct_tile_autotuning | 0.950× | 0.844× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | tma_loads | Δ | add_eviction_poli... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 48670 ns | 57540 ns | +8870 ns (+18.2%) | 58980 ns | +10310 ns (+21.2%) | 57950 ns | +9280 ns (+19.1%) | 57660 ns | +8990 ns (+18.5%) |
| SM Throughput (%) | 12.50 % | 19.83 % | +7.33 pp | 19.84 % | +7.34 pp | 19.82 % | +7.32 pp | 19.87 % | +7.37 pp |
| Memory Throughput (%) | 85.19 % | 76.35 % | -8.84 pp | 75.76 % | -9.43 pp | 76.28 % | -8.91 pp | 76.48 % | -8.71 pp |
| DRAM Throughput (%) | 1.65 % | 1.39 % | -0.26 pp | 1.36 % | -0.29 pp | 1.38 % | -0.27 pp | 1.39 % | -0.26 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 128.00 register/thread | +34 | 128.00 register/thread | +34 | 128.00 register/thread | +34 |
| Waves/SM | 0.19 | 0.23 | +0.04 | 0.23 | +0.04 | 0.23 | +0.04 | 0.23 | +0.04 |
| No Eligible (%) | 86.11 % | 77.95 % | -8.16 pp | 78.07 % | -8.04 pp | 77.94 % | -8.17 pp | 77.97 % | -8.14 pp |
| IPC Active | 0.55 inst/cycle | 0.88 inst/cycle | +0.33 | 0.88 inst/cycle | +0.33 | 0.88 inst/cycle | +0.33 | 0.88 inst/cycle | +0.33 |


### Key Learnings

#### ❌ Plan A: `fix_b_matrix_transposed_load_to_eliminate_bank_conflicts` (0.846×)

Rewriting the B-matrix load from a strided [BLOCK_K, BLOCK_N] layout to a contiguous [BLOCK_N, BLOCK_K] layout followed by tl.trans(b) in tl.dot produced no measurable improvement (0.8458x, effectively a wash or slight regression). All profiling metrics remained nearly identical before and after: SharedMemoryConflicts still reported 12-way bank conflicts on 77.65% of wavefronts, MIO stalls remained at ~63.7% of warp cycles, L1/TEX throughput stayed at ~84.6%, and executed instruction count was unchanged at 9.06M. The bank conflict count (6,666,240) was identical in both profiles, indicating the transposed load did not change the shared memory staging pattern that Triton uses internally for tl.dot. The root cause of the bank conflicts appears to be in how Triton stages operands into shared memory for tl.dot internally, not in the user-visible load layout.

> **Applicable when**: Never: this specific approach (changing user-visible load layout + tl.trans) does not resolve Triton-internal shared memory bank conflicts for tl.dot operands.

> **Avoid when**: When SharedMemoryConflicts is the primary bottleneck in a Triton kernel using tl.dot. Changing the user-visible load shape and using tl.trans() does not affect how Triton internally stages operands into shared memory for the matrix multiply unit. The bank conflict count (6,666,240) was identical before and after this change, confirming the approach is ineffective for this bottleneck.

#### ❌ Plan C: `add_eviction_policies_streaming_A_reused_B` (0.840×)

Adding eviction_policy='evict_first' to streaming loads and 'evict_last' to reused weight loads produced a 16% regression (0.84x speedup, 57.73us → 57.95us). All key metrics were essentially unchanged: L2 Hit Rate barely moved (68.95% → 68.92%), Mem Busy stayed flat (76.29% → 76.28%), and instruction count was identical (~9.06M). The bottleneck is L1/TEX cache saturation (84.6%) caused by 12-way shared memory bank conflicts consuming 77.65% of all shared load wavefronts — eviction policy hints operate on L2 and DRAM, not on shared memory, so they cannot address the structural bank conflict root cause. When the dominant stall is MIO/shared-memory bank conflicts (CPIStall rule, 63.7% of stall cycles), L2 eviction policy changes are ineffective and add marginal overhead.

> **Applicable when**: Never for this kernel configuration. L2 eviction policy hints are only effective when DRAM throughput is the bottleneck (high DRAM Throughput %, low L2 Hit Rate) and the data reuse pattern is clear. They are ineffective when L1/TEX cache throughput is saturated by shared memory bank conflicts.

> **Avoid when**: When SharedMemoryConflicts rule is triggered with >10-way bank conflicts and >75% conflicted wavefronts, and L1/TEX Cache Throughput exceeds 80%. In this regime, the bottleneck is shared memory access pattern (MIO stalls), not L2/DRAM bandwidth. Eviction policy hints (evict_first/evict_last) only affect L2 cache behavior and cannot reduce shared memory bank conflicts. DRAM Throughput must be meaningfully high (>20%) for eviction policies to have any impact.

#### ❌ Plan D: `fix_b_matrix_load_index_to_enable_correct_tile_autotuning` (0.844×)

Correcting the B matrix global memory load index (from `xindex = idx_n + 80*idx_m` to `xindex = idx_m + 480*idx_n`) and removing the redundant `tl.broadcast_to` wrapper produced no measurable speedup (0.8441x, effectively neutral within noise). All profiling metrics remained virtually identical: duration 57.73us vs 57.66us, bank conflicts unchanged at 6,666,240 (77.65% of wavefronts), MIO stall fraction unchanged at ~63.7%, occupancy unchanged at 6.25%, and executed instruction count identical at 9,061,152. The index fix alone — without also changing tile sizes via autotuning — does not alter the shared memory access pattern that causes the 12-way bank conflicts, so the structural bottleneck persists. The index correction is a necessary prerequisite for correctness when tile sizes change, but it is not itself a performance optimization.

> **Applicable when**: Never as a standalone optimization. The index fix is a correctness prerequisite when tile sizes (BLOCK_K, BLOCK_N) are changed via autotuning, but it produces no performance benefit by itself when tile sizes remain fixed.

> **Avoid when**: When tile sizes are not being changed simultaneously. Fixing only the load index without also changing BLOCK_M/BLOCK_N/BLOCK_K leaves the shared memory bank conflict pattern (12-way, 77.65% of wavefronts) completely unchanged, as confirmed by identical post-fix profiling metrics.

---

## Cumulative Progress

```
Baseline:  48670 ns  (1.000×)

Final best: 48700 ns  (0.9994×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -87,8 +101,6 @@

     xindex = idx_n + 80*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
