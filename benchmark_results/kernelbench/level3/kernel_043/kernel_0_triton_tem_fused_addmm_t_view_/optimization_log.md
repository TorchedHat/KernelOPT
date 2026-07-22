# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12370000 ns — **Best latency**: 12060000 ns (**1.0257×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`baseline`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_0_triton_tem_fused_addmm_t_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9216,1,1), Waves/SM: 34.91
- **Registers/thread**: 255
- **Duration**: 12370000 ns
- **SM Throughput**: 41.42 % | **Memory**: 60.46 % | **DRAM**: 1.64 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 12370000 ns | 12410000 ns | +40000 ns (+0.3%) |
| SM Throughput (%) | 41.42 % | 41.42 % | +0.00 pp |
| Memory Throughput (%) | 60.46 % | 60.47 % | +0.01 pp |
| DRAM Throughput (%) | 1.64 % | 1.66 % | +0.02 pp |
| Achieved Occupancy (%) | 12.46 % | 12.46 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 58.12 % | 58.11 % | -0.01 pp |
| IPC Active | 1.67 inst/cycle | 1.68 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_shared_memory_bank_conflicts_via_smem_padding | 1.030× | 1.000× | ✓ | **✓ best** |
| C | eviction_policies_streaming_a_reuse_b | 1.000× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12370000 ns | 12370000 ns | +0 ns (+0.0%) | 12360000 ns | -10000 ns (-0.1%) |
| SM Throughput (%) | 41.42 % | 41.52 % | +0.10 pp | 41.50 % | +0.08 pp |
| Memory Throughput (%) | 60.46 % | 60.61 % | +0.15 pp | 60.59 % | +0.13 pp |
| DRAM Throughput (%) | 1.64 % | 1.67 % | +0.03 pp | 1.48 % | -0.16 pp |
| Achieved Occupancy (%) | 12.46 % | 12.46 % | +0.00 pp | 12.46 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 58.12 % | 58.19 % | +0.07 pp | 58.08 % | -0.04 pp |
| IPC Active | 1.67 inst/cycle | 1.68 inst/cycle | +0.01 | 1.67 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_correct_b_transpose_via_block_ptr | 1.020× | 1.011× | ✓ |  |
| C | fix_b_load_broadcast_then_autotune_tile_sizes_for_occupancy | 1.028× | 0.998× | ✓ |  |
| E | add_triton_autotune | 1.030× | 1.033× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_correct... | Δ | fix_b_load_broadc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12370000 ns | 12240000 ns | -130000 ns (-1.1%) | 12390000 ns | +20000 ns (+0.2%) | 11980000 ns | -390000 ns (-3.2%) |
| SM Throughput (%) | 41.42 % | 41.73 % | +0.31 pp | 41.54 % | +0.12 pp | 42.75 % | +1.33 pp |
| Memory Throughput (%) | 60.46 % | 60.24 % | -0.22 pp | 60.66 % | +0.20 pp | 62.02 % | +1.56 pp |
| DRAM Throughput (%) | 1.64 % | 1.38 % | -0.26 pp | 1.65 % | +0.01 pp | 1.65 % | +0.01 pp |
| Achieved Occupancy (%) | 12.46 % | 12.48 % | +0.02 pp | 12.46 % | +0.00 pp | 12.48 % | +0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 58.12 % | 57.74 % | -0.38 pp | 58.15 % | +0.03 pp | 57.10 % | -1.02 pp |
| IPC Active | 1.67 inst/cycle | 1.69 inst/cycle | +0.02 | 1.67 inst/cycle | +0.00 | 1.71 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 1.0326× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 5

**Best so far**: 1.0326× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | increase_block_k_to_reduce_loop_overhead_and_improve_arithmetic_intensity | 0.930× | 0.244× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 12370000 ns | 50640000 ns | +38270000 ns (+309.4%) |
| SM Throughput (%) | 41.42 % | 12.31 % | -29.11 pp |
| Memory Throughput (%) | 60.46 % | 25.19 % | -35.27 pp |
| DRAM Throughput (%) | 1.64 % | 21.62 % | +19.98 pp |
| Achieved Occupancy (%) | 12.46 % | 12.47 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 58.12 % | 87.67 % | +29.55 pp |
| IPC Active | 1.67 inst/cycle | 0.49 inst/cycle | -1.18 |


### Key Learnings

#### ❌ Plan C: `increase_block_k_to_reduce_loop_overhead_and_improve_arithmetic_intensity` (0.244×)

Increasing BLOCK_K from 32 to 64 in a register-pressure-limited Triton GEMM kernel (255 registers/thread, 12.5% occupancy) caused a 4x slowdown. The larger K-tile doubled the A-tile footprint from 4KB to 8KB, collapsing the L1 hit rate from 54.9% to 5.55% and increasing L2 traffic by ~11x (L2 input sectors: 122M → 1336M). This triggered severe L1TEX scoreboard stalls (CPIStall: 9.2 cycles/instruction, 56.8% of total stall cycles), pushing warp cycles per issued instruction from 4.66 to 16.19 and dropping eligible warps/scheduler from 0.79 to 0.15. When occupancy is already at its minimum (2 warps/scheduler), increasing tile sizes that evict L1 data eliminates the only remaining latency-hiding mechanism, causing catastrophic stall amplification rather than the expected arithmetic intensity improvement.

> **Applicable when**: Never applicable in this configuration. Increasing BLOCK_K is harmful when: (1) the kernel is already register-limited to minimum occupancy (≤2 warps/scheduler), (2) L1 hit rate is moderate (50-60%) indicating the current tile already stresses L1 capacity, and (3) there is no slack in warp count to absorb additional memory latency.

> **Avoid when**: Do not increase BLOCK_K when theoretical occupancy is ≤12.5% (register-limited to 255 regs/thread) and active warps per scheduler is ≤2.0. In this regime, larger tiles destroy L1 hit rate (54.9% → 5.55%), multiply L2 sector traffic by ~11x, and amplify warp stall cycles from 4.66 to 16.19 cycles/instruction — a 4x wall-clock regression. The strategy is also inapplicable when the B-matrix load formula must not be changed (correctness constraint).

---

## Cumulative Progress

```
Baseline:  12370000 ns  (1.000×)
Iter 3     11980000 ns  (1.033×)  — add_triton_autotune

Final best: 12060000 ns  (1.0257×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_t_view_0(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -65,12 +75,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 768*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 2304*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')
 
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +95,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 2304*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
