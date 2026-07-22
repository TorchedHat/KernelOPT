# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3460 ns — **Best latency**: 3170 ns (**1.0915×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_lambda_and_add_low_warp_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_14_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (896,1,1), Waves/SM: 0.42
- **Registers/thread**: 28
- **Duration**: 3460 ns
- **SM Throughput**: 15.26 % | **Memory**: 13.80 % | **DRAM**: 1.56 %
- **Achieved Occupancy**: 34.10 %
- **IPC Active**: 1.41 inst/cycle | **No Eligible**: 59.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_sizes_and_add_autotune_for_transpose_kernel | 1.037× | 1.258× | ✓ |  |
| C | increase_tile_sizes_and_autotune_for_underutilized_transpose_kernel | 1.058× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 2750 ns | -710 ns (-20.5%) | 3460 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 15.26 % | 7.99 % | -7.27 pp | 12.24 % | -3.02 pp |
| Memory Throughput (%) | 13.80 % | 3.50 % | -10.30 pp | 5.85 % | -7.95 pp |
| DRAM Throughput (%) | 1.56 % | 1.95 % | +0.39 pp | 1.55 % | -0.01 pp |
| Achieved Occupancy (%) | 34.10 % | 28.17 % | -5.93 pp | 29.88 % | -4.22 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 28.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.42 | +0.00 |
| No Eligible (%) | 59.25 % | 67.74 % | +8.49 pp | 51.20 % | -8.05 pp |
| IPC Active | 1.41 inst/cycle | 1.41 inst/cycle | +0.00 | 1.98 inst/cycle | +0.57 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_underutilized_transpose_kernel | 0.056× | 0.028× | ✓ |  |
| B | coalesced_transpose_via_column_loop_unroll | 1.000× | 0.820× | ✓ |  |
| C | persistent_tile_scheduling_for_transpose_kernel | 0.011× | 0.006× | ✓ |  |
| D | fix_broken_autotune_grid_dispatch_for_transpose_kernel | 1.034× | 1.091× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | coalesced_transpo... | Δ | persistent_tile_s... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 123520 ns | +120060 ns (+3469.9%) | 4220 ns | +760 ns (+22.0%) | 613700 ns | +610240 ns (+17637.0%) | 3170 ns | -290 ns (-8.4%) |
| SM Throughput (%) | 15.26 % | 24.16 % | +8.90 pp | 46.66 % | +31.40 pp | 54.80 % | +39.54 pp | 12.29 % | -2.97 pp |
| Memory Throughput (%) | 13.80 % | 47.13 % | +33.33 pp | 26.72 % | +12.92 pp | 73.05 % | +59.25 pp | 5.87 % | -7.93 pp |
| DRAM Throughput (%) | 1.56 % | 0.05 % | -1.51 pp | 1.28 % | -0.28 pp | 0.01 % | -1.55 pp | 1.70 % | +0.14 pp |
| Achieved Occupancy (%) | 34.10 % | 41.73 % | +7.63 pp | 56.27 % | +22.17 pp | 42.39 % | +8.29 pp | 29.71 % | -4.39 pp |
| Registers/Thread | 28.00 register/thread | 22.00 register/thread | -6 | 20.00 register/thread | -8 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.85 | +0.43 | 0.42 | +0.00 | 0.42 | +0.00 |
| No Eligible (%) | 59.25 % | 72.76 % | +13.51 pp | 18.69 % | -40.56 pp | 43.62 % | -15.63 pp | 44.08 % | -15.17 pp |
| IPC Active | 1.41 inst/cycle | 1.10 inst/cycle | -0.31 | 3.18 inst/cycle | +1.77 | 2.31 inst/cycle | +0.90 | 1.97 inst/cycle | +0.56 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_dispatch_and_expand_warp_configs | 1.051× | 1.116× | ✓ |  |
| B | revert_static_range_regression_restore_vectorized_load_store_with_correct_autotune | 1.000× | 1.189× | ✓ |  |
| C | revert_persistent_loop_restore_2d_dispatch_with_correct_autotune_grid | 1.045× | 1.227× | ✓ |  |
| D | remove_persistent_loop_and_autotune_2d_tile_dispatch | 1.060× | 1.214× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.011× | 0.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | revert_static_ran... | Δ | revert_persistent... | Δ | remove_persistent... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 3100 ns | -360 ns (-10.4%) | 2910 ns | -550 ns (-15.9%) | 2820 ns | -640 ns (-18.5%) | 2850 ns | -610 ns (-17.6%) | 605920 ns | +602460 ns (+17412.1%) |
| SM Throughput (%) | 15.26 % | 14.90 % | -0.36 pp | 6.16 % | -9.10 pp | 7.93 % | -7.33 pp | 8.65 % | -6.61 pp | 58.00 % | +42.74 pp |
| Memory Throughput (%) | 13.80 % | 6.09 % | -7.71 pp | 3.41 % | -10.39 pp | 3.49 % | -10.31 pp | 3.71 % | -10.09 pp | 72.08 % | +58.28 pp |
| DRAM Throughput (%) | 1.56 % | 1.72 % | +0.16 pp | 1.85 % | +0.29 pp | 1.92 % | +0.36 pp | 1.90 % | +0.34 pp | 0.01 % | -1.55 pp |
| Achieved Occupancy (%) | 34.10 % | 60.84 % | +26.74 pp | 26.56 % | -7.54 pp | 30.76 % | -3.34 pp | 28.14 % | -5.96 pp | 16.94 % | -17.16 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 20.00 register/thread | -8 | 20.00 register/thread | -8 | 20.00 register/thread | -8 | 166.00 register/thread | +138 |
| Waves/SM | 0.42 | 0.85 | +0.43 | 0.42 | +0.00 | 0.42 | +0.00 | 0.42 | +0.00 | 2.26 | +1.84 |
| No Eligible (%) | 59.25 % | 41.61 % | -17.64 pp | 61.53 % | +2.28 pp | 56.66 % | -2.59 pp | 67.91 % | +8.66 pp | 39.77 % | -19.48 pp |
| IPC Active | 1.41 inst/cycle | 2.09 inst/cycle | +0.68 | 1.26 inst/cycle | -0.15 | 1.29 inst/cycle | -0.12 | 1.50 inst/cycle | +0.09 | 2.41 inst/cycle | +1.00 |

---

## Iteration 4

**Best so far**: 1.214× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_expand_autotune_warp_configs_for_transpose | 1.017× | 1.227× | ✓ |  |
| B | fix_grid_lambda_and_expand_warp_configs_for_autotune | 1.030× | 1.201× | ✓ |  |
| C | fix_grid_lambda_to_use_meta_tile_sizes_for_correct_autotune_dispatch | 0.981× | 1.407× | ✓ |  |
| D | fix_hardcoded_grid_and_add_num_stages_to_autotune_configs | 1.085× | 1.061× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 2820 ns | -640 ns (-18.5%) | 2880 ns | -580 ns (-16.8%) | 2460 ns | -1000 ns (-28.9%) | 3260 ns | -200 ns (-5.8%) |
| SM Throughput (%) | 15.26 % | 6.38 % | -8.88 pp | 8.39 % | -6.87 pp | 2.96 % | -12.30 pp | 13.67 % | -1.59 pp |
| Memory Throughput (%) | 13.80 % | 3.51 % | -10.29 pp | 3.62 % | -10.18 pp | 2.19 % | -11.61 pp | 5.62 % | -8.18 pp |
| DRAM Throughput (%) | 1.56 % | 1.92 % | +0.36 pp | 1.88 % | +0.32 pp | 2.19 % | +0.63 pp | 1.65 % | +0.09 pp |
| Achieved Occupancy (%) | 34.10 % | 26.74 % | -7.36 pp | 28.23 % | -5.87 pp | 12.28 % | -21.82 pp | 60.60 % | +26.50 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 20.00 register/thread | -8 | 16.00 register/thread | -12 | 20.00 register/thread | -8 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.42 | +0.00 | 0.21 | -0.21 | 0.85 | +0.43 |
| No Eligible (%) | 59.25 % | 61.41 % | +2.16 pp | 56.59 % | -2.66 pp | 89.34 % | +30.09 pp | 44.38 % | -14.87 pp |
| IPC Active | 1.41 inst/cycle | 1.23 inst/cycle | -0.18 | 1.52 inst/cycle | +0.11 | 0.36 inst/cycle | -1.05 | 2.11 inst/cycle | +0.70 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_and_expand_autotune_warp_configs_for_transpose` (1.227×)

Adding num_warps=8 variants and num_stages=2 to an existing autotune config set for a 2D Triton transpose kernel caused the autotuner to select a configuration that reduced executed instructions by ~21% (172K → 136K), yielding a 1.227x speedup despite the wall-clock duration remaining nearly identical (2.82 µs). The causal chain is: expanded config space → autotuner selects a different tile config (likely smaller XBLOCK or different YBLOCK) → fewer total instructions executed per launch. Notably, achieved occupancy slightly decreased (30.76% → 26.74%) and warp cycles per issued instruction increased (11.05 → 19.98), suggesting the winning config trades occupancy for fewer total instructions — a net win for this latency-bound small kernel. A new CPIStall rule appeared in the after profile (50.6% of cycles stalled on IMC miss), indicating the selected config accesses more distinct constant-cache addresses per warp.

> **Applicable when**: 2D Triton transpose/permute kernel where: (1) the autotune config space is limited to num_warps in {2, 4} only; (2) IssueSlotUtilization and AchievedOccupancy rules fire with no-eligible-warp% > 50%; (3) the kernel is small and latency-bound (waves_per_sm < 1, duration < 5 µs); (4) executed instruction count is the dominant cost metric. Expanding to num_warps=8 variants gives the autotuner more freedom to find a lower-instruction config.

> **Avoid when**: Do not apply when: (1) the kernel is already register-pressure-limited (registers_per_thread >= 28) — adding num_warps=8 may increase register pressure further; (2) the kernel is memory-bandwidth-bound (DRAM throughput > 50%) rather than instruction-count-bound — warp count changes will not help; (3) the CPIStall on IMC miss is already a dominant bottleneck before the change — expanding configs may select a config that worsens IMC stalls (as seen in the after profile here). Note: the speedup here (1.227x) comes from instruction reduction, not occupancy improvement — occupancy actually decreased.

---

## Iteration 5

**Best so far**: 1.214× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_hardcoded_grid_and_add_num_warps16_configs_for_autotune | 1.028× | 1.258× | ✓ |  |
| C | eliminate_imc_stall_via_constexpr_numel_and_fix_grid_lambda | 1.008× | 1.214× | ✓ |  |
| D | fix_hardcoded_grid_lambda_and_add_low_warp_autotune_configs | 1.090× | 1.091× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | eliminate_imc_sta... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 2750 ns | -710 ns (-20.5%) | 2850 ns | -610 ns (-17.6%) | 3170 ns | -290 ns (-8.4%) |
| SM Throughput (%) | 15.26 % | 8.00 % | -7.26 pp | 6.48 % | -8.78 pp | 14.81 % | -0.45 pp |
| Memory Throughput (%) | 13.80 % | 3.44 % | -10.36 pp | 3.56 % | -10.24 pp | 6.08 % | -7.72 pp |
| DRAM Throughput (%) | 1.56 % | 1.96 % | +0.40 pp | 1.89 % | +0.33 pp | 1.71 % | +0.15 pp |
| Achieved Occupancy (%) | 34.10 % | 28.23 % | -5.87 pp | 26.70 % | -7.40 pp | 63.58 % | +29.48 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 20.00 register/thread | -8 | 20.00 register/thread | -8 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.42 | +0.00 | 0.85 | +0.43 |
| No Eligible (%) | 59.25 % | 55.68 % | -3.57 pp | 69.79 % | +10.54 pp | 42.35 % | -16.90 pp |
| IPC Active | 1.41 inst/cycle | 1.48 inst/cycle | +0.07 | 1.27 inst/cycle | -0.14 | 1.85 inst/cycle | +0.44 |

---

## Cumulative Progress

```
Baseline:  3460 ns  (1.000×)
Iter 3     2850 ns  (1.214×)  — remove_persistent_loop_and_autotune_2d_tile_dispatch

Final best: 3170 ns  (1.0915×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1},  num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 1},  num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 1},  num_warps=8, num_stages=3),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 128,  'XBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 64,   'XBLOCK': 1},  num_warps=2, num_stages=2),
+        triton.Config({'YBLOCK': 64,   'XBLOCK': 1},  num_warps=1, num_stages=2),
+        triton.Config({'YBLOCK': 32,   'XBLOCK': 1},  num_warps=1, num_stages=2),
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 128,  'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 64,   'XBLOCK': 16}, num_warps=2, num_stages=2),
+        triton.Config({'YBLOCK': 64,   'XBLOCK': 16}, num_warps=1, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 7168
-    xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 224)
-    y1 = yindex // 224
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 224*x2 + 2016*y1), tmp0, xmask)
-
-
+    y0 = y3 % 224
+    y1 = y3 // 224
+    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 224 * x2 + 2016 * y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +49,7 @@

     ynumel = 7168
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
