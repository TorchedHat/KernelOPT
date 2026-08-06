# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_78` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 49540 ns — **Best latency**: 47870 ns (**1.0349×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_78`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_78_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 94
- **Duration**: 49540 ns
- **SM Throughput**: 7.92 % | **Memory**: 53.11 % | **DRAM**: 0.90 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 1.033× | 1.035× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_reshape | 0.996× | 1.000× | ✓ |  |
| D | autotune_tile_sizes_to_increase_grid_and_reduce_bank_conflicts | 0.482× | 0.467× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 49540 ns | 47870 ns | -1670 ns (-3.4%) | 49540 ns | +0 ns (+0.0%) | 106180 ns | +56640 ns (+114.3%) |
| SM Throughput (%) | 7.92 % | 7.92 % | +0.00 pp | 7.93 % | +0.01 pp | 11.78 % | +3.86 pp |
| Memory Throughput (%) | 53.11 % | 51.19 % | -1.92 pp | 53.38 % | +0.27 pp | 45.98 % | -7.13 pp |
| DRAM Throughput (%) | 0.90 % | 0.93 % | +0.03 pp | 0.90 % | +0.00 pp | 0.42 % | -0.48 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 64.00 register/thread | -30 | 94.00 register/thread | +0 | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.08 | -0.12 | 0.20 | +0.00 | 0.20 | +0.00 |
| No Eligible (%) | 86.40 % | 86.41 % | +0.01 pp | 86.38 % | -0.02 pp | 79.65 % | -6.75 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.55 inst/cycle | +0.00 | 0.81 inst/cycle | +0.26 |

---

## Iteration 2

**Best so far**: 1.0349× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | promote_tile_sizes_to_autotune_meta_parameters_fix_bank_conflicts | 0.483× | 0.466× | ✓ |  |
| B | promote_tile_sizes_to_autotune_meta_parameters_with_dynamic_grid | 0.470× | 0.429× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.449× | 0.247× | ✓ |  |


### Metric Comparison

| Metric | Baseline | promote_tile_size... | Δ | promote_tile_size... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 49540 ns | 106270 ns | +56730 ns (+114.5%) | 115360 ns | +65820 ns (+132.9%) | 200160 ns | +150620 ns (+304.0%) |
| SM Throughput (%) | 7.92 % | 11.82 % | +3.90 pp | 11.70 % | +3.78 pp | 10.99 % | +3.07 pp |
| Memory Throughput (%) | 53.11 % | 46.17 % | -6.94 pp | 45.86 % | -7.25 pp | 35.92 % | -17.19 pp |
| DRAM Throughput (%) | 0.90 % | 0.42 % | -0.48 pp | 0.40 % | -0.50 pp | 0.23 % | -0.67 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 208.00 register/thread | +114 | 64.00 register/thread | -30 |
| Waves/SM | 0.20 | 0.20 | +0.00 | 0.61 | +0.41 | 0.20 | +0.00 |
| No Eligible (%) | 86.40 % | 79.66 % | -6.74 pp | 79.38 % | -7.02 pp | 85.06 % | -1.34 pp |
| IPC Active | 0.55 inst/cycle | 0.81 inst/cycle | +0.26 | 0.82 inst/cycle | +0.27 | 0.60 inst/cycle | +0.05 |


### Key Learnings

#### ❌ Plan B: `promote_tile_sizes_to_autotune_meta_parameters_with_dynamic_grid` (0.429×)

Increasing BLOCK_M and BLOCK_N from 32 to 64 (and BLOCK_K from 32 to 128) caused registers per thread to balloon from 64 to 208 and dynamic shared memory per block to jump from 16 KB to 131 KB. This collapsed Block Limit Shared Mem from 9 to 1 and Block Limit Registers from 8 to 2, dropping theoretical occupancy from 50% to 6.25% — eliminating any latency-hiding capacity. The instruction count also tripled (3M → 10.7M), likely due to a B-matrix load rewrite that introduced incorrect or redundant indexing, resulting in a 2.33x slowdown despite the bank conflict stall cycles per instruction improving slightly (7.32 → 4.83 cycles). The root cause is that the diff did NOT promote tile sizes to autotune meta-parameters in Config dicts — instead it hard-coded larger constexprs in the kernel body while also rewriting the B-matrix load pattern, causing massive register spill and smem pressure.

> **Applicable when**: Never apply this specific pattern — this is a negative example. The anti-pattern is: hard-coding larger tile sizes as constexprs in the kernel body (not in autotune Config dicts) while simultaneously rewriting the B-matrix load with explicit pointer arithmetic, causing register count to triple and shared memory per block to jump 8x.

> **Avoid when**: Always avoid hard-coding large tile sizes (BLOCK_M=64, BLOCK_N=64, BLOCK_K=128) as kernel-body constexprs when the kernel already has high register pressure (64+ registers/thread). Before increasing tile sizes, verify that the resulting shared memory per block (BLOCK_M*BLOCK_K*4 + BLOCK_K*BLOCK_N*4 bytes) still allows at least 2 blocks per SM. If Block Limit Shared Mem would drop to 1, the tile size increase is counterproductive regardless of bank conflict improvements.

---

## Iteration 3

**Best so far**: 1.0349× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_stride_to_eliminate_shared_memory_bank_conflicts | 0.481× | 0.465× | ✓ |  |
| B | reduce_tile_sizes_to_recover_smem_occupancy_and_grid_coverage | 1.000× | 0.999× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_via_proper_autotune_meta_parameters | 0.483× | 0.466× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | reduce_tile_sizes... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 49540 ns | 106620 ns | +57080 ns (+115.2%) | 49600 ns | +60 ns (+0.1%) | 106340 ns | +56800 ns (+114.7%) |
| SM Throughput (%) | 7.92 % | 11.80 % | +3.88 pp | 7.90 % | -0.02 pp | 11.78 % | +3.86 pp |
| Memory Throughput (%) | 53.11 % | 46.06 % | -7.05 pp | 53.19 % | +0.08 pp | 46.01 % | -7.10 pp |
| DRAM Throughput (%) | 0.90 % | 0.42 % | -0.48 pp | 0.90 % | +0.00 pp | 0.42 % | -0.48 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 94.00 register/thread | +0 | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.20 | +0.00 | 0.20 | +0.00 | 0.20 | +0.00 |
| No Eligible (%) | 86.40 % | 79.73 % | -6.67 pp | 86.37 % | -0.03 pp | 79.65 % | -6.75 pp |
| IPC Active | 0.55 inst/cycle | 0.81 inst/cycle | +0.26 | 0.54 inst/cycle | -0.01 | 0.81 inst/cycle | +0.26 |

---

## Cumulative Progress

```
Baseline:  49540 ns  (1.000×)
Iter 1     47870 ns  (1.035×)  — autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts

Final best: 47870 ns  (1.0349×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_78(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_M : tl.constexpr = 64
+    BLOCK_N : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -92,8 +104,6 @@

     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
