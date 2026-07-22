# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_45` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4540 ns — **Best latency**: 3840 ns (**1.1823×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_tile_dimensions_to_eliminate_partial_wave_and_thread_waste`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_45`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_45_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 28
- **Duration**: 4540 ns
- **SM Throughput**: 36.93 % | **Memory**: 32.58 % | **DRAM**: 4.08 %
- **Achieved Occupancy**: 69.30 %
- **IPC Active**: 2.70 inst/cycle | **No Eligible**: 32.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_dimensions_to_eliminate_partial_wave_and_thread_waste | 1.089× | 1.182× | ✓ | **✓ best** |
| D | fix_tile_dimensions_eliminate_partial_wave_and_improve_coalescing | 1.022× | 1.051× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_dimensio... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 3840 ns | -700 ns (-15.4%) | 4320 ns | -220 ns (-4.8%) |
| SM Throughput (%) | 36.93 % | 6.60 % | -30.33 pp | 15.42 % | -21.51 pp |
| Memory Throughput (%) | 32.58 % | 4.83 % | -27.75 pp | 11.26 % | -21.32 pp |
| DRAM Throughput (%) | 4.08 % | 4.83 % | +0.75 pp | 4.30 % | +0.22 pp |
| Achieved Occupancy (%) | 69.30 % | 31.87 % | -37.43 pp | 47.61 % | -21.69 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 22.00 register/thread | -6 |
| Waves/SM | 1.48 | 0.74 | -0.74 | 0.74 | -0.74 |
| No Eligible (%) | 32.06 % | 69.10 % | +37.04 pp | 41.24 % | +9.18 pp |
| IPC Active | 2.70 inst/cycle | 0.92 inst/cycle | -1.78 | 2.02 inst/cycle | -0.68 |

---

## Iteration 2

**Best so far**: 1.1823× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_underutilized_transpose_kernel | 0.006× | 0.002× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotune_lambda_and_add_large_yblock_configs | 1.079× | 1.173× | ✓ |  |
| C | persistent_tile_scheduling_multi_tile_per_sm | 0.001× | 0.001× | ✓ |  |
| D | increase_yblock_tile_size_to_fill_gpu_waves_and_improve_occupancy | 0.977× | 1.051× | ✓ |  |
| E | add_triton_autotune | 1.078× | 1.173× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_hardcoded_gri... | Δ | persistent_tile_s... | Δ | increase_yblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 2840000 ns | +2835460 ns (+62455.1%) | 3870 ns | -670 ns (-14.8%) | 6300000 ns | +6295460 ns (+138666.5%) | 4320 ns | -220 ns (-4.8%) | 3870 ns | -670 ns (-14.8%) |
| SM Throughput (%) | 36.93 % | 71.70 % | +34.77 pp | 6.60 % | -30.33 pp | 68.63 % | +31.70 pp | 15.43 % | -21.50 pp | 6.60 % | -30.33 pp |
| Memory Throughput (%) | 32.58 % | 21.27 % | -11.31 pp | 4.82 % | -27.76 pp | 88.77 % | +56.19 pp | 11.31 % | -21.27 pp | 4.81 % | -27.77 pp |
| DRAM Throughput (%) | 4.08 % | 0.01 % | -4.07 pp | 4.82 % | +0.74 pp | 0.01 % | -4.07 pp | 4.32 % | +0.24 pp | 4.81 % | +0.73 pp |
| Achieved Occupancy (%) | 69.30 % | 50.03 % | -19.27 pp | 26.18 % | -43.12 pp | 79.67 % | +10.37 pp | 48.34 % | -20.96 pp | 26.40 % | -42.90 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 22.00 register/thread | -6 | 16.00 register/thread | -12 |
| Waves/SM | 1.48 | 0.74 | -0.74 | 0.74 | -0.74 | 1.48 | +0.00 | 0.74 | -0.74 | 0.74 | -0.74 |
| No Eligible (%) | 32.06 % | 27.16 % | -4.90 pp | 68.33 % | +36.27 pp | 30.42 % | -1.64 pp | 42.58 % | +10.52 pp | 67.96 % | +35.90 pp |
| IPC Active | 2.70 inst/cycle | 2.89 inst/cycle | +0.19 | 1.02 inst/cycle | -1.68 | 2.78 inst/cycle | +0.08 | 2.08 inst/cycle | -0.62 | 1.01 inst/cycle | -1.69 |


### Key Learnings

#### ❌ Plan C: `persistent_tile_scheduling_multi_tile_per_sm` (0.001×)

Converting a short-lived Triton kernel (3.87µs, 190K instructions) to a persistent while-loop kernel while leaving the launcher grid unchanged (still (cdiv(xnumel,128), cdiv(ynumel,8)) = (1, 3136)) caused a ~1440x increase in elapsed cycles (5748 → 8,376,417) and a ~17,200x explosion in executed instructions (190K → 3.275B). The persistent loop body ran correctly but was executed by all 3136 CTAs instead of the intended 132 SMs, multiplying total work by 3136/132 ≈ 24x per tile iteration. Although compute throughput rose from 6.6% to 74.05% and L1 hit rate improved from 47% to 94.74%, these are artifacts of the kernel doing vastly more redundant work — the improvements are meaningless. The persistent kernel pattern is an anti-pattern for sub-10µs kernels and is catastrophically broken when the launcher grid is not simultaneously changed to (NUM_SMS,).

> **Applicable when**: Never apply this pattern. This is a documented anti-pattern: persistent while-loop kernel with the original 2D grid formula retained instead of being changed to (NUM_SMS,).

> **Avoid when**: Always avoid persistent kernels on sub-10µs workloads with <1M instructions. If a persistent kernel is attempted, the launcher grid MUST be changed to (NUM_SMS,) simultaneously — the kernel body change alone is insufficient and causes catastrophic work multiplication proportional to old_grid_size / NUM_SMS.

---

## Iteration 3

**Best so far**: 1.1823× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launcher_grid_lambda_and_add_num_stages_pipelining | 0.006× | 0.001× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotune_lambda_enabling_real_tile_exploration | 0.989× | 1.044× | ✓ |  |
| C | fix_hardcoded_grid_and_add_num_stages_to_pipeline_uncoalesced_stores | 0.006× | 0.001× | ✓ |  |
| D | fix_grid_launch_to_use_persistent_1d_sm_count_grid | 0.006× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launcher_grid... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 5580000 ns | +5575460 ns (+122807.5%) | 4350 ns | -190 ns (-4.2%) | 5590000 ns | +5585460 ns (+123027.8%) | 2850000 ns | +2845460 ns (+62675.3%) |
| SM Throughput (%) | 36.93 % | 74.05 % | +37.12 pp | 15.34 % | -21.59 pp | 74.07 % | +37.14 pp | 71.56 % | +34.63 pp |
| Memory Throughput (%) | 32.58 % | 11.45 % | -21.13 pp | 11.36 % | -21.22 pp | 11.46 % | -21.12 pp | 21.18 % | -11.40 pp |
| DRAM Throughput (%) | 4.08 % | 0.01 % | -4.07 pp | 4.29 % | +0.21 pp | 0.01 % | -4.07 pp | 0.01 % | -4.07 pp |
| Achieved Occupancy (%) | 69.30 % | 49.62 % | -19.68 pp | 48.57 % | -20.73 pp | 49.73 % | -19.57 pp | 50.13 % | -19.17 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 22.00 register/thread | -6 | 20.00 register/thread | -8 | 20.00 register/thread | -8 |
| Waves/SM | 1.48 | 0.74 | -0.74 | 0.74 | -0.74 | 0.74 | -0.74 | 0.74 | -0.74 |
| No Eligible (%) | 32.06 % | 25.05 % | -7.01 pp | 44.11 % | +12.05 pp | 25.03 % | -7.03 pp | 27.21 % | -4.85 pp |
| IPC Active | 2.70 inst/cycle | 2.97 inst/cycle | +0.27 | 2.07 inst/cycle | -0.63 | 2.97 inst/cycle | +0.27 | 2.89 inst/cycle | +0.19 |

---

## Cumulative Progress

```
Baseline:  4540 ns  (1.000×)
Iter 1     3840 ns  (1.182×)  — fix_tile_dimensions_to_eliminate_partial_wave_and_thread_waste

Final best: 3840 ns  (1.1823×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 2, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 2, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 2, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 25088
@@ -22,15 +43,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 112*x2 + 1008*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 25088
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_45[grid](arg_0, arg_1, 25088, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_45[grid](arg_0, arg_1, 25088, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
