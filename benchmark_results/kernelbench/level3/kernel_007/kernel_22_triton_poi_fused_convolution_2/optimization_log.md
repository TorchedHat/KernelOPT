# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4540 ns — **Best latency**: 4160 ns (**1.0913×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_static_grid_to_lambda_grid_and_add_num_stages_to_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_22`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_22_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 1.45
- **Registers/thread**: 28
- **Duration**: 4540 ns
- **SM Throughput**: 34.78 % | **Memory**: 34.12 % | **DRAM**: 4.01 %
- **Achieved Occupancy**: 67.43 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 39.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_configuration_eliminate_partial_wave | 0.963× | 0.445× | ✓ |  |
| B | fix_tile_sizes_eliminate_partial_wave | 0.970× | 1.007× | ✓ |  |
| C | autotune_tile_sizes_fix_partial_wave_and_coalescing | 0.793× | 0.713× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_config... | Δ | fix_tile_sizes_el... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 10210 ns | +5670 ns (+124.9%) | 4510 ns | -30 ns (-0.7%) | 6370 ns | +1830 ns (+40.3%) |
| SM Throughput (%) | 34.78 % | 65.63 % | +30.85 pp | 33.12 % | -1.66 pp | 57.38 % | +22.60 pp |
| Memory Throughput (%) | 34.12 % | 63.29 % | +29.17 pp | 32.38 % | -1.74 pp | 48.83 % | +14.71 pp |
| DRAM Throughput (%) | 4.01 % | 1.79 % | -2.22 pp | 4.02 % | +0.01 pp | 2.88 % | -1.13 pp |
| Achieved Occupancy (%) | 67.43 % | 52.31 % | -15.12 pp | 71.71 % | +4.28 pp | 71.90 % | +4.47 pp |
| Registers/Thread | 28.00 register/thread | 48.00 register/thread | +20 | 28.00 register/thread | +0 | 32.00 register/thread | +4 |
| Waves/SM | 1.45 | 2.33 | +0.88 | 1.45 | +0.00 | 1.45 | +0.00 |
| No Eligible (%) | 39.58 % | 17.53 % | -22.05 pp | 32.78 % | -6.80 pp | 17.12 % | -22.46 pp |
| IPC Active | 2.48 inst/cycle | 3.28 inst/cycle | +0.80 | 2.47 inst/cycle | -0.01 | 3.29 inst/cycle | +0.81 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_flatten_1d | 0.004× | 0.003× | ✓ |  |
| B | persistent_tile_scheduling_flatten_2d_to_1d | 0.000× | 0.000× | ✓ |  |
| C | fix_autotune_grid_lambda_to_match_tile_sizes | 1.000× | 0.448× | ✓ |  |
| D | add_triton_autotune | 1.003× | 1.007× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 1620000 ns | +1615460 ns (+35582.8%) | 37430000 ns | +37425460 ns (+824349.3%) | 10140 ns | +5600 ns (+123.3%) | 4510 ns | -30 ns (-0.7%) |
| SM Throughput (%) | 34.78 % | 41.02 % | +6.24 pp | 74.52 % | +39.74 pp | 66.06 % | +31.28 pp | 34.49 % | -0.29 pp |
| Memory Throughput (%) | 34.12 % | 66.50 % | +32.38 pp | 23.93 % | -10.19 pp | 63.69 % | +29.57 pp | 33.86 % | -0.26 pp |
| DRAM Throughput (%) | 4.01 % | 0.02 % | -3.99 pp | 0.00 % | -4.01 pp | 1.80 % | -2.21 pp | 4.05 % | +0.04 pp |
| Achieved Occupancy (%) | 67.43 % | 98.19 % | +30.76 pp | 73.64 % | +6.21 pp | 52.51 % | -14.92 pp | 68.88 % | +1.45 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 18.00 register/thread | -10 | 48.00 register/thread | +20 | 28.00 register/thread | +0 |
| Waves/SM | 1.45 | 1.45 | +0.00 | 1.45 | +0.00 | 2.33 | +0.88 | 1.45 | +0.00 |
| No Eligible (%) | 39.58 % | 44.01 % | +4.43 pp | 22.95 % | -16.63 pp | 17.67 % | -21.91 pp | 35.34 % | -4.24 pp |
| IPC Active | 2.48 inst/cycle | 2.23 inst/cycle | -0.25 | 3.07 inst/cycle | +0.59 | 3.21 inst/cycle | +0.73 | 2.52 inst/cycle | +0.04 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_flatten_1d` (0.003×)

Converting a short-running 2D tiled kernel (4.51µs, 1.23M instructions, 1.45 waves) to a persistent 1D kernel with an internal while-loop caused a catastrophic 432x instruction explosion (1.23M → 531M instructions) and a ~324x duration increase (4.51µs → 1.46ms). Although occupancy rose from 68.9% to 97.1% and L1 hit rate rose from 28% to 96.8%, these gains were completely overwhelmed by the instruction count explosion; CPIStall fired with ~78% of cycles stalled on L1TEX scoreboard dependencies and eligible warps per scheduler collapsed to 1.05. The persistent kernel pattern is catastrophically wrong for tiny kernels with only 1–2 waves — the partial-wave tail penalty (≤50%) is far smaller than the instruction explosion from persistent looping with non-power-of-2 stride divisions.

> **Applicable when**: Never for kernels with ≤2 waves and sub-10µs duration. Persistent kernels only help when the kernel has many waves (>>10) and per-tile compute is substantial enough to amortize loop overhead.

> **Avoid when**: Do NOT apply persistent kernel patterns to short-running kernels (< 10µs, < 5 waves). The while-loop overhead and integer divide/modulo per iteration cause instruction count to explode relative to the original grid-based dispatch. This regression was reproduced across two consecutive iterations (iter 2 and iter 3) with nearly identical metrics, confirming the pattern is robustly harmful for this kernel class.

#### ❌ Plan B: `persistent_tile_scheduling_flatten_2d_to_1d` (0.000×)

Converting a 2D-tiled Triton kernel to a persistent 1D while-loop kernel caused a ~2600x regression (4.51µs → 11.74ms) with a 3594x instruction explosion (1.23M → 4.42B instructions). Even with a hardcoded numel=221184 constant inside the kernel, the while-loop with stride=num_programs*XBLOCK caused each of the 3072 CTAs to iterate over the entire element space, executing vastly more total work than the original single-tile-per-CTA design. L1 hit rate improved dramatically (28% → 95%) and occupancy rose (68.9% → 87.6%), but these gains were completely overwhelmed by the instruction explosion; CPIStall fired with 49.3% of cycles stalled on L1TEX scoreboard dependencies. Persistent while-loop kernels are catastrophically wrong for short-running kernels (< 10µs, ≤ 2 waves) — the loop overhead dominates all other effects.

> **Applicable when**: Never for kernels with ≤2 waves and sub-10µs duration. Persistent kernels only help when the kernel has many waves (>>10) and per-tile compute is substantial enough to amortize loop overhead.

> **Avoid when**: Do NOT apply persistent while-loop patterns to short-running kernels (< 10µs, < 5 waves). Even with a hardcoded numel constant, the while-loop causes each CTA to iterate over the full element space, exploding instruction count. The partial-wave tail penalty (≤50%) is far smaller than the instruction explosion from persistent looping. Also avoid 1D flattening of permuted-layout store kernels — the non-linear store address (y0 + 128*x2 + 1152*y1) creates uncoalesced writes when iterating linearly over the flat index.

---

## Iteration 3

**Best so far**: 1.0067× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_configs_xblock_oversized_for_xnumel9 | 0.001× | 0.000× | ✓ |  |
| B | fix_autotune_configs_reduce_register_pressure_and_fix_grid_lambda | 1.018× | 1.076× | ✓ | **✓ best** |
| C | revert_broken_persistent_loop_to_correct_2d_tiled_kernel | 0.967× | 1.013× | ✓ |  |
| D | fix_broken_persistent_kernel_rewrite_as_correct_2d_tiled | 0.940× | 0.991× | ✓ |  |
| E | add_triton_autotune | 0.004× | 0.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_conf... | Δ | fix_autotune_conf... | Δ | revert_broken_per... | Δ | fix_broken_persis... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 11740000 ns | +11735460 ns (+258490.3%) | 4220 ns | -320 ns (-7.0%) | 4480 ns | -60 ns (-1.3%) | 4580 ns | +40 ns (+0.9%) | 1460000 ns | +1455460 ns (+32058.6%) |
| SM Throughput (%) | 34.78 % | 47.68 % | +12.90 pp | 20.59 % | -14.19 pp | 28.96 % | -5.82 pp | 29.02 % | -5.76 pp | 41.64 % | +6.86 pp |
| Memory Throughput (%) | 34.12 % | 29.64 % | -4.48 pp | 19.45 % | -14.67 pp | 33.26 % | -0.86 pp | 33.38 % | -0.74 pp | 67.49 % | +33.37 pp |
| DRAM Throughput (%) | 4.01 % | 0.01 % | -4.00 pp | 4.31 % | +0.30 pp | 4.07 % | +0.06 pp | 3.99 % | -0.02 pp | 0.02 % | -3.99 pp |
| Achieved Occupancy (%) | 67.43 % | 87.56 % | +20.13 pp | 49.62 % | -17.81 pp | 70.10 % | +2.67 pp | 69.91 % | +2.48 pp | 97.11 % | +29.68 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 1.45 | 1.45 | +0.00 | 0.73 | -0.72 | 1.45 | +0.00 | 1.45 | +0.00 | 1.45 | +0.00 |
| No Eligible (%) | 39.58 % | 38.98 % | -0.60 pp | 37.64 % | -1.94 pp | 40.38 % | +0.80 pp | 40.82 % | +1.24 pp | 43.62 % | +4.04 pp |
| IPC Active | 2.48 inst/cycle | 2.48 inst/cycle | +0.00 | 2.37 inst/cycle | -0.11 | 2.20 inst/cycle | -0.28 | 2.29 inst/cycle | -0.19 | 2.20 inst/cycle | -0.28 |

---

## Iteration 4

**Best so far**: 1.0758× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_latency_hiding_and_scheduler_utilization | 0.982× | 1.066× | ✓ |  |
| B | autotune_xblock16_yblock_sweep_correct_grid_lambda | 1.021× | 1.091× | ✓ |  |
| D | revert_persistent_loop_to_2d_tiled_kernel_with_correct_grid_lambda | 1.025× | 1.091× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | autotune_xblock16... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 4260 ns | -280 ns (-6.2%) | 4160 ns | -380 ns (-8.4%) | 4160 ns | -380 ns (-8.4%) |
| SM Throughput (%) | 34.78 % | 20.86 % | -13.92 pp | 19.44 % | -15.34 pp | 19.16 % | -15.62 pp |
| Memory Throughput (%) | 34.12 % | 19.48 % | -14.64 pp | 20.56 % | -13.56 pp | 20.26 % | -13.86 pp |
| DRAM Throughput (%) | 4.01 % | 4.29 % | +0.28 pp | 4.41 % | +0.40 pp | 4.39 % | +0.38 pp |
| Achieved Occupancy (%) | 67.43 % | 50.81 % | -16.62 pp | 48.41 % | -19.02 pp | 48.38 % | -19.05 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 1.45 | 0.73 | -0.72 | 0.73 | -0.72 | 0.73 | -0.72 |
| No Eligible (%) | 39.58 % | 40.28 % | +0.70 pp | 42.61 % | +3.03 pp | 45.07 % | +5.49 pp |
| IPC Active | 2.48 inst/cycle | 2.25 inst/cycle | -0.23 | 2.20 inst/cycle | -0.28 | 2.19 inst/cycle | -0.29 |


### Key Learnings

#### ✅ Plan D: `revert_persistent_loop_to_2d_tiled_kernel_with_correct_grid_lambda` (1.091×)

Reordering the xoffset/xindex/xmask computation before yoffset/yindex/ymask and combining them into a single `mask = xmask & ymask` (removing the inline `xmask & ymask` at each load/store site) reduced executed instructions by ~12% (714K → 628K) and warp cycles per issued instruction by 12.6% (15.51 → 13.55 cycles), yielding a 1.09x speedup. Removing the `eviction_policy='evict_last'` hint on the load also contributed by allowing the compiler to generate more efficient instruction sequences. The structural change is minor — same 2D tiled kernel, same grid, same tile sizes — but the reordering and mask consolidation reduced compiler-generated instruction count.

> **Applicable when**: 2D tiled Triton kernels where the mask is a conjunction of two independent dimension masks (xmask & ymask) and the load uses eviction_policy hints. Warp cycles per issued instruction is elevated (>12) and instruction count is higher than expected for the tile size.

> **Avoid when**: Kernels where eviction_policy='evict_last' is intentionally set for L1 cache management (e.g., streaming access patterns where eviction policy meaningfully affects cache hit rate). Also not applicable when the mask is not separable into independent x/y components.

---

## Iteration 5

**Best so far**: 1.0913× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_grid_and_add_num_stages_to_autotune | 0.989× | 1.091× | ✓ |  |
| B | fix_static_grid_to_lambda_grid_and_expand_yblock_configs_for_wave_balance | 0.991× | 1.066× | ✓ |  |
| C | add_num_stages_software_pipelining_and_remove_eviction_policy | 0.994× | 1.107× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 4160 ns | -380 ns (-8.4%) | 4260 ns | -280 ns (-6.2%) | 4100 ns | -440 ns (-9.7%) |
| SM Throughput (%) | 34.78 % | 19.54 % | -15.24 pp | 18.92 % | -15.86 pp | 18.02 % | -16.76 pp |
| Memory Throughput (%) | 34.12 % | 20.66 % | -13.46 pp | 19.95 % | -14.17 pp | 20.77 % | -13.35 pp |
| DRAM Throughput (%) | 4.01 % | 4.38 % | +0.37 pp | 4.27 % | +0.26 pp | 4.46 % | +0.45 pp |
| Achieved Occupancy (%) | 67.43 % | 48.40 % | -19.03 pp | 48.58 % | -18.85 pp | 48.12 % | -19.31 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 1.45 | 0.73 | -0.72 | 0.73 | -0.72 | 0.73 | -0.72 |
| No Eligible (%) | 39.58 % | 42.30 % | +2.72 pp | 41.49 % | +1.91 pp | 45.30 % | +5.72 pp |
| IPC Active | 2.48 inst/cycle | 2.18 inst/cycle | -0.30 | 2.18 inst/cycle | -0.30 | 2.07 inst/cycle | -0.41 |

---

## Cumulative Progress

```
Baseline:  4540 ns  (1.000×)
Iter 2     4510 ns  (1.007×)  — add_triton_autotune
Iter 3     4220 ns  (1.076×)  — fix_autotune_configs_reduce_register_pressure_and_fix_grid_lambda
Iter 4     4160 ns  (1.091×)  — revert_persistent_loop_to_2d_tiled_kernel_with_correct_grid_lambda

Final best: 4160 ns  (1.0913×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 16}, num_warps=2),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 24576
-    xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
+    mask = ymask & xmask
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    y0 = y3 % 128
+    y1 = y3 // 128
+    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask)
+    tl.store(out_ptr0 + (y0 + 128 * x2 + 1152 * y1), tmp0, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +39,7 @@

     ynumel = 24576
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_22[grid](arg_0, arg_1, 24576, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_22[grid](arg_0, arg_1, 24576, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
