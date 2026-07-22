# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_group_norm_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 281380 ns — **Best latency**: 260130 ns (**1.0817×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reorder_pid_decomposition_for_coalesced_output_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_group_norm_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_085/kernel_2_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.94
- **Registers/thread**: 61
- **Duration**: 281380 ns
- **SM Throughput**: 53.38 % | **Memory**: 90.52 % | **DRAM**: 41.74 %
- **Achieved Occupancy**: 46.89 %
- **IPC Active**: 2.25 inst/cycle | **No Eligible**: 43.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_via_xblock_increase_and_autotune | 1.160× | 0.994× | ✓ | **✓ best** |
| B | software_pipelining_num_stages | 0.033× | 0.025× | ✓ |  |
| C | increase_xblock_and_autotune_tile_sizes | 0.171× | 0.003× | ✓ |  |
| D | autotune_xblock_r0block_to_fix_partial_wave_and_coalescing | 1.040× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | software_pipelini... | Δ | increase_xblock_a... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281380 ns | 283230 ns | +1850 ns (+0.7%) | 11430000 ns | +11148620 ns (+3962.1%) | 90640000 ns | +90358620 ns (+32112.7%) | 283200 ns | +1820 ns (+0.6%) |
| SM Throughput (%) | 53.38 % | 54.49 % | +1.11 pp | 1.35 % | -52.03 pp | 14.00 % | -39.38 pp | 54.01 % | +0.63 pp |
| Memory Throughput (%) | 90.52 % | 89.96 % | -0.56 pp | 4.06 % | -86.46 pp | 67.18 % | -23.34 pp | 89.87 % | -0.65 pp |
| DRAM Throughput (%) | 41.74 % | 41.82 % | +0.08 pp | 1.09 % | -40.65 pp | 67.18 % | +25.44 pp | 42.26 % | +0.52 pp |
| Achieved Occupancy (%) | 46.89 % | 47.02 % | +0.13 pp | 39.99 % | -6.90 pp | 96.56 % | +49.67 pp | 46.92 % | +0.03 pp |
| Registers/Thread | 61.00 register/thread | 61.00 register/thread | +0 | 65.00 register/thread | +4 | 32.00 register/thread | -29 | 61.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 2.22 | +0.28 | 0.97 | -0.97 | 1.94 | +0.00 |
| No Eligible (%) | 43.24 % | 43.31 % | +0.07 pp | 98.62 % | +55.38 pp | 94.33 % | +51.09 pp | 43.77 % | +0.53 pp |
| IPC Active | 2.25 inst/cycle | 2.24 inst/cycle | -0.01 | 0.06 inst/cycle | -2.19 | 0.23 inst/cycle | -2.02 | 2.28 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_output_stores_via_xblock16_and_fixed_grid_lambda | 1.007× | 0.100× | ✓ |  |
| B | fix_autotune_grid_lambda_for_coalesced_output_stores | 1.155× | 0.099× | ✓ |  |
| C | fix_strided_global_access_and_remove_mio_saturating_num_stages | 1.159× | 0.977× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_blocks_and_register_spill | 0.171× | 0.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_output_... | Δ | fix_autotune_grid... | Δ | fix_strided_globa... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281380 ns | 2820000 ns | +2538620 ns (+902.2%) | 2840000 ns | +2558620 ns (+909.3%) | 287940 ns | +6560 ns (+2.3%) | 90610000 ns | +90328620 ns (+32102.0%) |
| SM Throughput (%) | 53.38 % | 75.85 % | +22.47 pp | 72.45 % | +19.07 pp | 53.21 % | -0.17 pp | 14.01 % | -39.37 pp |
| Memory Throughput (%) | 90.52 % | 12.06 % | -78.46 pp | 12.61 % | -77.91 pp | 88.53 % | -1.99 pp | 67.20 % | -23.32 pp |
| DRAM Throughput (%) | 41.74 % | 3.77 % | -37.97 pp | 3.74 % | -38.00 pp | 42.71 % | +0.97 pp | 67.20 % | +25.46 pp |
| Achieved Occupancy (%) | 46.89 % | 36.53 % | -10.36 pp | 12.20 % | -34.69 pp | 46.92 % | +0.03 pp | 96.56 % | +49.67 pp |
| Registers/Thread | 61.00 register/thread | 70.00 register/thread | +9 | 255.00 register/thread | +194 | 61.00 register/thread | +0 | 32.00 register/thread | -29 |
| Waves/SM | 1.94 | 2.22 | +0.28 | 7.76 | +5.82 | 1.94 | +0.00 | 0.97 | -0.97 |
| No Eligible (%) | 43.24 % | 21.76 % | -21.48 pp | 25.37 % | -17.87 pp | 43.25 % | +0.01 pp | 94.32 % | +51.08 pp |
| IPC Active | 2.25 inst/cycle | 3.13 inst/cycle | +0.88 | 2.98 inst/cycle | +0.73 | 2.24 inst/cycle | -0.01 | 0.23 inst/cycle | -2.02 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_eliminate_partial_wave_and_improve_occupancy | 1.175× | 1.076× | ✓ |  |
| B | fix_grid_lambda_and_reduce_register_pressure_via_small_xblock | 1.143× | 0.995× | ✓ |  |
| C | restore_small_xblock_configs_and_fix_grid_lambda_to_eliminate_register_spill | 1.249× | 1.127× | ✓ | **✓ best** |
| D | fix_grid_lambda_and_constrain_autotune_to_safe_xblock | 1.105× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | fix_grid_lambda_a... | Δ | restore_small_xbl... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281380 ns | 261570 ns | -19810 ns (-7.0%) | 282940 ns | +1560 ns (+0.6%) | 249730 ns | -31650 ns (-11.2%) | 285860 ns | +4480 ns (+1.6%) |
| SM Throughput (%) | 53.38 % | 58.01 % | +4.63 pp | 51.64 % | -1.74 pp | 52.79 % | -0.59 pp | 53.19 % | -0.19 pp |
| Memory Throughput (%) | 90.52 % | 90.06 % | -0.46 pp | 93.03 % | +2.51 pp | 92.68 % | +2.16 pp | 89.74 % | -0.78 pp |
| DRAM Throughput (%) | 41.74 % | 40.91 % | -0.83 pp | 37.78 % | -3.96 pp | 42.63 % | +0.89 pp | 41.93 % | +0.19 pp |
| Achieved Occupancy (%) | 46.89 % | 45.49 % | -1.40 pp | 23.81 % | -23.08 pp | 95.13 % | +48.24 pp | 46.94 % | +0.05 pp |
| Registers/Thread | 61.00 register/thread | 61.00 register/thread | +0 | 128.00 register/thread | +67 | 32.00 register/thread | -29 | 61.00 register/thread | +0 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 3.88 | +1.94 | 0.97 | -0.97 | 1.94 | +0.00 |
| No Eligible (%) | 43.24 % | 39.92 % | -3.32 pp | 46.44 % | +3.20 pp | 44.55 % | +1.31 pp | 43.39 % | +0.15 pp |
| IPC Active | 2.25 inst/cycle | 2.39 inst/cycle | +0.14 | 2.15 inst/cycle | -0.10 | 2.23 inst/cycle | -0.02 | 2.24 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.1267× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_locality_via_pid_reorder_group_norm_reduction | 1.251× | 1.132× | ✓ |  |
| B | restrict_xblock1_and_expand_r0block_num_warps_to_reduce_register_pressure | 1.290× | 1.082× | ✓ | **✓ best** |
| C | restrict_autotune_to_xblock1_to_eliminate_register_pressure_and_restore_occupancy | 1.289× | 1.077× | ✓ |  |
| D | restrict_autotune_to_xblock1_to_reduce_register_pressure_and_maximize_occupancy | 1.252× | 1.133× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_locality_via_p... | Δ | restrict_xblock1_... | Δ | restrict_autotune... | Δ | restrict_autotune... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281380 ns | 248540 ns | -32840 ns (-11.7%) | 260130 ns | -21250 ns (-7.6%) | 261250 ns | -20130 ns (-7.2%) | 248380 ns | -33000 ns (-11.7%) |
| SM Throughput (%) | 53.38 % | 52.81 % | -0.57 pp | 57.55 % | +4.17 pp | 56.77 % | +3.39 pp | 52.50 % | -0.88 pp |
| Memory Throughput (%) | 90.52 % | 93.05 % | +2.53 pp | 90.44 % | -0.08 pp | 89.99 % | -0.53 pp | 93.49 % | +2.97 pp |
| DRAM Throughput (%) | 41.74 % | 42.81 % | +1.07 pp | 41.07 % | -0.67 pp | 40.93 % | -0.81 pp | 42.82 % | +1.08 pp |
| Achieved Occupancy (%) | 46.89 % | 95.13 % | +48.24 pp | 45.54 % | -1.35 pp | 45.49 % | -1.40 pp | 95.11 % | +48.22 pp |
| Registers/Thread | 61.00 register/thread | 32.00 register/thread | -29 | 61.00 register/thread | +0 | 61.00 register/thread | +0 | 32.00 register/thread | -29 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 0.97 | -0.97 | 0.97 | -0.97 | 0.97 | -0.97 |
| No Eligible (%) | 43.24 % | 44.42 % | +1.18 pp | 39.03 % | -4.21 pp | 39.29 % | -3.95 pp | 44.37 % | +1.13 pp |
| IPC Active | 2.25 inst/cycle | 2.20 inst/cycle | -0.05 | 2.38 inst/cycle | +0.13 | 2.39 inst/cycle | +0.14 | 2.22 inst/cycle | -0.03 |

---

## Iteration 5

**Best so far**: 1.1267× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_false_max_contiguous_hint_on_strided_load_offset | 1.251× | 1.125× | ✓ |  |
| C | reorder_pid_decomposition_for_coalesced_output_stores | 1.281× | 1.082× | ✓ |  |
| D | cap_register_pressure_via_maxnreg_to_double_occupancy | 1.287× | 1.090× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_false_max_con... | Δ | reorder_pid_decom... | Δ | cap_register_pres... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 281380 ns | 250140 ns | -31240 ns (-11.1%) | 260130 ns | -21250 ns (-7.6%) | 258110 ns | -23270 ns (-8.3%) |
| SM Throughput (%) | 53.38 % | 52.72 % | -0.66 pp | 57.75 % | +4.37 pp | 57.53 % | +4.15 pp |
| Memory Throughput (%) | 90.52 % | 92.49 % | +1.97 pp | 90.59 % | +0.07 pp | 91.10 % | +0.58 pp |
| DRAM Throughput (%) | 41.74 % | 42.54 % | +0.80 pp | 41.20 % | -0.54 pp | 41.41 % | -0.33 pp |
| Achieved Occupancy (%) | 46.89 % | 95.20 % | +48.31 pp | 45.39 % | -1.50 pp | 45.43 % | -1.46 pp |
| Registers/Thread | 61.00 register/thread | 32.00 register/thread | -29 | 61.00 register/thread | +0 | 61.00 register/thread | +0 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 0.97 | -0.97 | 0.97 | -0.97 |
| No Eligible (%) | 43.24 % | 44.53 % | +1.29 pp | 39.56 % | -3.68 pp | 39.24 % | -4.00 pp |
| IPC Active | 2.25 inst/cycle | 2.21 inst/cycle | -0.04 | 2.39 inst/cycle | +0.14 | 2.40 inst/cycle | +0.15 |


### Key Learnings

#### ✅ Plan A: `fix_false_max_contiguous_hint_on_strided_load_offset` (1.125×)

Replacing an incorrect tl.max_contiguous(tl.multiple_of(offset, [1,4]), [1,4]) hint on a stride-64 access pattern with the correct tl.multiple_of(offset, [1,64]) annotation yielded a 1.12x speedup, but the key NCU metrics (load bytes utilized per sector: 16/32, excessive sectors: 16.3M, Mem Busy: ~93%) remained essentially unchanged before and after. The profiling data shows no measurable improvement in memory access efficiency, suggesting the Triton compiler did not materially change its load instruction generation based on this hint correction, and the speedup likely came from minor microarchitectural or scheduling effects rather than the intended coalescing improvement. When the dominant bottleneck is uncoalesced global stores (4/32 bytes/sector, 81% potential speedup per NCU) and uncoalesced loads (16/32 bytes/sector, 45% potential speedup), correcting compiler hints alone is insufficient — the underlying memory layout must change.

> **Applicable when**: Triton kernels where a tl.max_contiguous hint is applied to an offset that has a large stride (e.g., stride-64) between consecutive reduction elements, and NCU reports MemoryCacheAccessPattern with load bytes utilized per sector significantly below 32 (e.g., 16/32) combined with UncoalescedGlobalAccess. The hint correction is always semantically correct and safe to apply.

> **Avoid when**: When the dominant bottleneck is uncoalesced global stores (4/32 bytes/sector) caused by output layout — hint corrections cannot fix layout-inherent access patterns. Also do not expect large speedups from hint corrections alone when the NCU metrics show the bottleneck is in the store path (81% potential speedup from stores vs. 45% from loads); the store pattern requires output tensor layout restructuring.

---

## Cumulative Progress

```
Baseline:  281380 ns  (1.000×)
Iter 3     249730 ns  (1.127×)  — restore_small_xblock_configs_and_fix_grid_lambda_to_eliminate_register_spill

Final best: 260130 ns  (1.0817×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK':  512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 8192}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -50,7 +61,6 @@

     tl.store(out_ptr0 + (x4), tmp4, mask=xmask)
     tl.store(out_ptr1 + (x4), tmp8, mask=xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -59,7 +69,7 @@

     xnumel = 2048
     r0_numel = 63504
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, 2048, 63504, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused_convolution_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, 2048, 63504)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
