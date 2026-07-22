# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9980 ns — **Best latency**: 10180 ns (**0.9804×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_9_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (11760,1,1), Waves/SM: 5.57
- **Registers/thread**: 23
- **Duration**: 9980 ns
- **SM Throughput**: 55.61 % | **Memory**: 33.10 % | **DRAM**: 12.31 %
- **Achieved Occupancy**: 74.49 %
- **IPC Active**: 2.82 inst/cycle | **No Eligible**: 27.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.992× | 0.987× | ✓ |  |
| B | add_autotune_xblock_exploration | 1.020× | 0.980× | ✓ | **✓ best** |
| C | autotune_xblock_vectorized_loads | 0.964× | 0.928× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_maxpool | 1.004× | 0.725× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | add_autotune_xblo... | Δ | autotune_xblock_v... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9980 ns | 10110 ns | +130 ns (+1.3%) | 10180 ns | +200 ns (+2.0%) | 10750 ns | +770 ns (+7.7%) | 13760 ns | +3780 ns (+37.9%) |
| SM Throughput (%) | 55.61 % | 56.10 % | +0.49 pp | 56.55 % | +0.94 pp | 61.64 % | +6.03 pp | 62.44 % | +6.83 pp |
| Memory Throughput (%) | 33.10 % | 32.75 % | -0.35 pp | 33.26 % | +0.16 pp | 40.10 % | +7.00 pp | 33.97 % | +0.87 pp |
| DRAM Throughput (%) | 12.31 % | 12.19 % | -0.12 pp | 12.07 % | -0.24 pp | 11.44 % | -0.87 pp | 8.95 % | -3.36 pp |
| Achieved Occupancy (%) | 74.49 % | 71.65 % | -2.84 pp | 70.44 % | -4.05 pp | 79.39 % | +4.90 pp | 61.60 % | -12.89 pp |
| Registers/Thread | 23.00 register/thread | 23.00 register/thread | +0 | 23.00 register/thread | +0 | 32.00 register/thread | +9 | 40.00 register/thread | +17 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 7.42 | +1.85 |
| No Eligible (%) | 27.75 % | 29.40 % | +1.65 pp | 28.27 % | +0.52 pp | 23.53 % | -4.22 pp | 26.03 % | -1.72 pp |
| IPC Active | 2.82 inst/cycle | 2.84 inst/cycle | +0.02 | 2.84 inst/cycle | +0.02 | 3.05 inst/cycle | +0.23 | 2.94 inst/cycle | +0.12 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_maxpool` (0.725×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the winning config to use 40 registers/thread (up from 23), which reduced theoretical occupancy from 100% to 75% and achieved occupancy from 74.5% to 61.6%. Although L1 hit rate improved dramatically (0.77% → 37.15%) and warp cycles per instruction dropped (16.05 → 13.16), the total executed instruction count increased by 55% and elapsed cycles grew from 14,603 to 20,236, resulting in a 38% slowdown. The register pressure introduced by larger tile sizes (more live values in flight) can cap theoretical occupancy below the baseline, negating any cache reuse benefit.

> **Applicable when**: Kernel has very low L1 hit rate (<5%) and CPIStall dominated by L1TEX latency, AND registers/thread is already low (<24) leaving headroom for larger tiles without hitting occupancy limits.

> **Avoid when**: Kernel already uses 20+ registers/thread; increasing XBLOCK to 512+ will push register count to 35–45, capping theoretical occupancy at 75% or below and causing net regression even if L1 hit rate improves. Also avoid when the kernel body contains many live intermediate values (e.g., 9 simultaneous tl.load results held in registers) that scale with XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_and_tune_num_warps_for_occupancy | 0.991× | 0.984× | ✓ |  |
| B | fix_autotune_grid_to_enable_config_selection | 1.014× | 0.990× | ✓ |  |
| C | reduce_num_warps_to_cut_cpi_not_selected_stalls | 1.014× | 0.990× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_cap | 0.999× | 0.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | reduce_num_warps_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9980 ns | 10140 ns | +160 ns (+1.6%) | 10080 ns | +100 ns (+1.0%) | 10080 ns | +100 ns (+1.0%) | 10180 ns | +200 ns (+2.0%) |
| SM Throughput (%) | 55.61 % | 56.01 % | +0.40 pp | 56.31 % | +0.70 pp | 31.88 % | -23.73 pp | 55.67 % | +0.06 pp |
| Memory Throughput (%) | 33.10 % | 32.93 % | -0.17 pp | 32.90 % | -0.20 pp | 31.86 % | -1.24 pp | 33.27 % | +0.17 pp |
| DRAM Throughput (%) | 12.31 % | 12.11 % | -0.20 pp | 12.21 % | -0.10 pp | 12.23 % | -0.08 pp | 12.12 % | -0.19 pp |
| Achieved Occupancy (%) | 74.49 % | 72.08 % | -2.41 pp | 74.61 % | +0.12 pp | 38.15 % | -36.34 pp | 75.00 % | +0.51 pp |
| Registers/Thread | 23.00 register/thread | 23.00 register/thread | +0 | 23.00 register/thread | +0 | 32.00 register/thread | +9 | 23.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 2.78 | -2.79 | 5.57 | +0.00 |
| No Eligible (%) | 27.75 % | 28.50 % | +0.75 pp | 28.33 % | +0.58 pp | 46.75 % | +19.00 pp | 28.25 % | +0.50 pp |
| IPC Active | 2.82 inst/cycle | 2.86 inst/cycle | +0.04 | 2.84 inst/cycle | +0.02 | 2.01 inst/cycle | -0.81 | 2.83 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_to_2d_tiling_channel_spatial_for_l1_reuse | 0.998× | 0.993× | ✓ |  |
| B | fix_autotune_grid_lambda_and_wave_align_xblock | 1.010× | 0.987× | ✓ |  |
| C | use_cg_cache_modifier_to_bypass_l1_and_improve_l2_reuse | 1.006× | 0.993× | ✓ |  |
| D | add_cache_modifier_l1_hint_to_neighborhood_loads | 1.001× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_to_2d... | Δ | fix_autotune_grid... | Δ | use_cg_cache_modi... | Δ | add_cache_modifie... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9980 ns | 10050 ns | +70 ns (+0.7%) | 10110 ns | +130 ns (+1.3%) | 10050 ns | +70 ns (+0.7%) | 9980 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 55.61 % | 56.57 % | +0.96 pp | 33.01 % | -22.60 pp | 56.69 % | +1.08 pp | 54.99 % | -0.62 pp |
| Memory Throughput (%) | 33.10 % | 33.74 % | +0.64 pp | 31.81 % | -1.29 pp | 34.14 % | +1.04 pp | 32.94 % | -0.16 pp |
| DRAM Throughput (%) | 12.31 % | 12.24 % | -0.07 pp | 12.15 % | -0.16 pp | 12.26 % | -0.05 pp | 12.31 % | +0.00 pp |
| Achieved Occupancy (%) | 74.49 % | 73.75 % | -0.74 pp | 38.46 % | -36.03 pp | 72.51 % | -1.98 pp | 71.40 % | -3.09 pp |
| Registers/Thread | 23.00 register/thread | 23.00 register/thread | +0 | 32.00 register/thread | +9 | 23.00 register/thread | +0 | 23.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 2.78 | -2.79 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 27.75 % | 29.01 % | +1.26 pp | 47.16 % | +19.41 pp | 28.16 % | +0.41 pp | 29.93 % | +2.18 pp |
| IPC Active | 2.82 inst/cycle | 2.80 inst/cycle | -0.02 | 2.03 inst/cycle | -0.79 | 2.85 inst/cycle | +0.03 | 2.84 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  9980 ns  (1.000×)

Final best: 10180 ns  (0.9804×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1505280
@@ -33,13 +46,12 @@

              tl.maximum(t6, tl.maximum(t7, t8))))))))
     tl.store(out_ptr0 + x6, result, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 1505280
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_9[grid](arg_0, arg_1, 1505280, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_9[grid](arg_0, arg_1, 1505280)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
