# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2780 ns — **Best latency**: 2940 ns (**0.9456×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_10_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (640,1,1), Waves/SM: 0.30
- **Registers/thread**: 28
- **Duration**: 2780 ns
- **SM Throughput**: 11.52 % | **Memory**: 10.49 % | **DRAM**: 1.40 %
- **Achieved Occupancy**: 25.67 %
- **IPC Active**: 1.24 inst/cycle | **No Eligible**: 67.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_grid_utilization | 1.082× | 1.158× | ✓ |  |
| B | autotune_tile_sizes_for_small_grid_utilization | 1.042× | 1.034× | ✓ |  |
| C | increase_tile_sizes_and_autotune_for_underutilized_transpose_kernel | 0.975× | 0.877× | ✓ |  |
| D | autotune_tile_sizes_fix_xblock_waste | 1.090× | 0.946× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | increase_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2400 ns | -380 ns (-13.7%) | 2690 ns | -90 ns (-3.2%) | 3170 ns | +390 ns (+14.0%) | 2940 ns | +160 ns (+5.8%) |
| SM Throughput (%) | 11.52 % | 1.96 % | -9.56 pp | 5.53 % | -5.99 pp | 16.77 % | +5.25 pp | 5.87 % | -5.65 pp |
| Memory Throughput (%) | 10.49 % | 1.60 % | -8.89 pp | 2.47 % | -8.02 pp | 20.92 % | +10.43 pp | 2.62 % | -7.87 pp |
| DRAM Throughput (%) | 1.40 % | 1.60 % | +0.20 pp | 1.44 % | +0.04 pp | 1.25 % | -0.15 pp | 1.32 % | -0.08 pp |
| Achieved Occupancy (%) | 25.67 % | 8.91 % | -16.76 pp | 23.59 % | -2.08 pp | 13.11 % | -12.56 pp | 22.06 % | -3.61 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 20.00 register/thread | -8 | 48.00 register/thread | +20 | 20.00 register/thread | -8 |
| Waves/SM | 0.30 | 0.15 | -0.15 | 0.30 | +0.00 | 0.24 | -0.06 | 0.30 | +0.00 |
| No Eligible (%) | 67.03 % | 87.25 % | +20.22 pp | 65.22 % | -1.81 pp | 54.67 % | -12.36 pp | 64.84 % | -2.19 pp |
| IPC Active | 1.24 inst/cycle | 0.31 inst/cycle | -0.93 | 1.01 inst/cycle | -0.23 | 1.62 inst/cycle | +0.38 | 1.17 inst/cycle | -0.07 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.103× | 0.025× | ✓ |  |
| C | persistent_tile_scheduling_fix_grid_lambda_for_autotune | 1.088× | 0.885× | ✓ |  |
| D | persistent_tile_scheduling_for_small_grid_transpose | 0.005× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 111810 ns | +109030 ns (+3921.9%) | 3140 ns | +360 ns (+12.9%) | 1560000 ns | +1557220 ns (+56015.1%) |
| SM Throughput (%) | 11.52 % | 25.49 % | +13.97 pp | 9.82 % | -1.70 pp | 40.63 % | +29.11 pp |
| Memory Throughput (%) | 10.49 % | 14.47 % | +3.98 pp | 4.15 % | -6.34 pp | 15.80 % | +5.31 pp |
| DRAM Throughput (%) | 1.40 % | 0.04 % | -1.36 pp | 1.23 % | -0.17 pp | 0.00 % | -1.40 pp |
| Achieved Occupancy (%) | 25.67 % | 15.03 % | -10.64 pp | 46.55 % | +20.88 pp | 30.30 % | +4.63 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 20.00 register/thread | -8 | 16.00 register/thread | -12 |
| Waves/SM | 0.30 | 0.15 | -0.15 | 0.61 | +0.31 | 0.30 | +0.00 |
| No Eligible (%) | 67.03 % | 73.99 % | +6.96 pp | 50.70 % | -16.33 pp | 63.84 % | -3.19 pp |
| IPC Active | 1.24 inst/cycle | 1.03 inst/cycle | -0.21 | 1.76 inst/cycle | +0.52 | 1.44 inst/cycle | +0.20 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.025×)

Adding a persistent while-loop stride pattern to a tiny Triton kernel (46080 elements, 640 CTAs, 0.15 waves) caused a 40x regression in duration (2.4us → 111.81us) and a 584x explosion in executed instructions (36K → 21.2M). The grid remained at 0.15 waves (264 CTAs were not actually launched — the grid lambda was not changed), so each of the 640 CTAs iterated through all tiles via the while-loop instead of just its own tile, processing every element ~640 times. The persistent loop pattern requires the grid to be reduced to num_sm*2 CTAs; if the grid stays at the original size, every CTA redundantly processes all tiles, multiplying work by the number of CTAs. The L1 hit rate jumped to 93% (data fits in cache) but the sheer instruction count overwhelmed any bandwidth savings.

> **Applicable when**: Never apply persistent stride loop without simultaneously reducing the launch grid to num_sm*2 (or similar). This is a structural anti-pattern when the grid size is not co-reduced.

> **Avoid when**: The launch grid is not simultaneously reduced to match the persistent CTA count. For very small kernels (total work < 1MB, elapsed_cycles < 10K), the overhead of the while-loop control flow and index arithmetic exceeds any SM utilization benefit. When waves_per_sm < 0.5 due to small total work rather than large tile sizes, persistent scheduling cannot help because there is simply not enough work to amortize the loop overhead.

#### ❌ Plan D: `persistent_tile_scheduling_for_small_grid_transpose` (0.002×)

The persistent kernel body was correctly rewritten to loop over tiles using tl.range(pid, num_tiles, num_ctas), but the launch grid was NOT updated — it remained hardcoded to (cdiv(9,128), cdiv(5120,8)) = (1, 640) instead of the intended (264,). As a result, num_ctas = tl.num_programs(0) = 1 (only one CTA in dim 0), causing a single CTA to serially iterate over all ceil(46080/8)=5760 tiles, executing 442M instructions vs. 36K before — a 12,000x instruction blowup. Duration exploded from 2.4µs to 1.56ms (650x slower). The anti-pattern is implementing a persistent loop in the kernel body while leaving the launch grid unchanged: the loop stride becomes 1 instead of NUM_PERSISTENT_CTAS, serializing all work onto one CTA.

> **Applicable when**: Never apply this negative pattern. This documents the failure mode: persistent kernel loop body implemented without updating the launch grid to match NUM_PERSISTENT_CTAS.

> **Avoid when**: Always ensure the launch grid is updated atomically with the persistent loop body. If the grid stays at the original 2D shape, the persistent loop degenerates to a serial single-CTA execution with catastrophic instruction blowup (12,000x in this case).

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_coalescing_configs | 1.074× | 1.061× | ✓ |  |
| C | revert_and_fix_2d_autotune_with_correct_grid_lambda | 1.079× | 1.045× | ✓ |  |
| D | add_triton_autotune | 1.045× | 0.853× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | revert_and_fix_2d... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2620 ns | -160 ns (-5.8%) | 2660 ns | -120 ns (-4.3%) | 3260 ns | +480 ns (+17.3%) |
| SM Throughput (%) | 11.52 % | 4.54 % | -6.98 pp | 2.20 % | -9.32 pp | 9.61 % | -1.91 pp |
| Memory Throughput (%) | 10.49 % | 3.80 % | -6.69 pp | 1.47 % | -9.02 pp | 4.04 % | -6.45 pp |
| DRAM Throughput (%) | 1.40 % | 1.49 % | +0.09 pp | 1.47 % | +0.07 pp | 1.19 % | -0.21 pp |
| Achieved Occupancy (%) | 25.67 % | 10.77 % | -14.90 pp | 8.78 % | -16.89 pp | 46.68 % | +21.01 pp |
| Registers/Thread | 28.00 register/thread | 21.00 register/thread | -7 | 16.00 register/thread | -12 | 20.00 register/thread | -8 |
| Waves/SM | 0.30 | 0.15 | -0.15 | 0.15 | -0.15 | 0.61 | +0.31 |
| No Eligible (%) | 67.03 % | 83.26 % | +16.23 pp | 86.43 % | +19.40 pp | 50.75 % | -16.28 pp |
| IPC Active | 1.24 inst/cycle | 0.59 inst/cycle | -0.65 | 0.32 inst/cycle | -0.92 | 1.77 inst/cycle | +0.53 |


### Key Learnings

#### ❌ Plan D: `add_triton_autotune` (0.853×)

Adding @triton.autotune with new XBLOCK configs (128, 256, 512, 1024) while leaving the launch grid hardcoded produced a 0.85x regression. The grid remained fixed at 640 CTAs (waves_per_sm=0.61 unchanged), so autotune's config selection had zero effect on actual tile coverage — the decorator was purely decorative overhead. The slight duration increase (3.14→3.26µs) and instruction count increase (259K→259K issued, nearly identical) indicate the autotune tuning overhead itself added latency without any compensating benefit. The anti-pattern is applying @triton.autotune without simultaneously converting the hardcoded grid to a lambda that reads meta['XBLOCK']/meta['YBLOCK']; without the grid lambda, autotune cannot change the number of CTAs launched and the kernel behavior is identical to the baseline except for added tuning overhead.

> **Applicable when**: Any Triton kernel where @triton.autotune is added but the launch grid is not simultaneously updated to a lambda using meta['BLOCK_SIZE'] or equivalent. This is always a regression risk when the grid is hardcoded.

> **Avoid when**: Do not add @triton.autotune without also updating the grid to a lambda. The two changes are inseparable: autotune without a grid lambda is strictly worse than the baseline due to tuning overhead with zero benefit.

---

## Cumulative Progress

```
Baseline:  2780 ns  (1.000×)

Final best: 2940 ns  (0.9456×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},   num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},   num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 16},   num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},   num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},   num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 16},   num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},   num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},   num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},   num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},   num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},   num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 5120
@@ -22,14 +45,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, mask=xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 5120
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10[grid](arg_0, arg_1, 5120, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10[grid](arg_0, arg_1, 5120, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
