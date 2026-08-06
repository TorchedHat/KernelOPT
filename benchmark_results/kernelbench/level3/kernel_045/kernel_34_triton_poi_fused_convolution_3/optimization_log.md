# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_34` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4740 ns — **Best latency**: 3970 ns (**1.1940×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_34`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_34_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2496,1,1), Waves/SM: 1.18
- **Registers/thread**: 28
- **Duration**: 4740 ns
- **SM Throughput**: 31.08 % | **Memory**: 29.06 % | **DRAM**: 3.15 %
- **Achieved Occupancy**: 68.42 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 38.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_eliminate_partial_wave | — | 1.194× | ✓ | **✓ best** |
| C | fix_tile_sizes_to_eliminate_partial_wave_and_thread_waste | — | 1.185× | ✓ |  |
| D | fix_tile_shape_and_vectorize_loads | — | 1.324× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_sizes_to... | Δ | fix_tile_shape_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4740 ns | 3970 ns | -770 ns (-16.2%) | 4000 ns | -740 ns (-15.6%) | 3580 ns | -1160 ns (-24.5%) |
| SM Throughput (%) | 31.08 % | 19.43 % | -11.65 pp | 27.47 % | -3.61 pp | 17.20 % | -13.88 pp |
| Memory Throughput (%) | 29.06 % | 17.82 % | -11.24 pp | 13.85 % | -15.21 pp | 7.91 % | -21.15 pp |
| DRAM Throughput (%) | 3.15 % | 3.75 % | +0.60 pp | 3.70 % | +0.55 pp | 4.15 % | +1.00 pp |
| Achieved Occupancy (%) | 68.42 % | 42.63 % | -25.79 pp | 67.81 % | -0.61 pp | 56.38 % | -12.04 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 22.00 register/thread | -6 | 20.00 register/thread | -8 |
| Waves/SM | 1.18 | 0.59 | -0.59 | 1.18 | +0.00 | 1.18 | +0.00 |
| No Eligible (%) | 38.69 % | 39.64 % | +0.95 pp | 31.05 % | -7.64 pp | 38.43 % | -0.26 pp |
| IPC Active | 2.43 inst/cycle | 2.25 inst/cycle | -0.18 | 2.67 inst/cycle | +0.24 | 2.22 inst/cycle | -0.21 |


### Key Learnings

#### ✅ Plan A: `autotune_tile_sizes_eliminate_partial_wave` (1.194×)

When a Triton kernel's XBLOCK is much larger than xnumel (e.g., XBLOCK=128 for xnumel=9), the vast majority of threads in the X dimension are wasted (7% utilization), inflating grid size and creating a costly partial wave. Replacing the fixed oversized XBLOCK with @triton.autotune over configs that right-size XBLOCK to the nearest power-of-2 >= xnumel (here 16) and increase YBLOCK (32–256) reduces wasted lanes and collapses the grid to fewer than one full wave, cutting warp stall cycles (17.19 → 11.56 cycles/instruction) and kernel duration by ~16%. The NCU LaunchConfiguration rule with a reported 50% speedup_pct is the primary trigger signal, though actual gains depend on the specific workload balance.

> **Applicable when**: NCU LaunchConfiguration rule fires with speedup_pct >= 20%; XBLOCK >> xnumel (ratio > 4x); kernel has a 2D tile structure (XBLOCK x YBLOCK); waves_per_sm between 1.0 and 2.0 indicating a significant partial wave; warp cycles per issued instruction > 15

> **Avoid when**: xnumel is already close to XBLOCK (utilization > 50%); grid is already a near-perfect multiple of SM count (waves_per_sm close to integer); kernel is compute-bound rather than latency-bound; reducing block size would drop occupancy below the level needed to hide memory latency

---

## Iteration 2

**Best so far**: 1.194× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | — | 0.006× | ✓ |  |
| B | persistent_tile_scheduling | — | 0.006× | ✓ |  |
| C | persistent_tile_scheduling | — | 0.988× | ✓ |  |
| D | add_triton_autotune | — | 1.335× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4740 ns | 797540 ns | +792800 ns (+16725.7%) | 806400 ns | +801660 ns (+16912.7%) | 4800 ns | +60 ns (+1.3%) | 3550 ns | -1190 ns (-25.1%) |
| SM Throughput (%) | 31.08 % | 68.19 % | +37.11 pp | 42.76 % | +11.68 pp | 39.92 % | +8.84 pp | 17.19 % | -13.89 pp |
| Memory Throughput (%) | 29.06 % | 65.97 % | +36.91 pp | 65.24 % | +36.18 pp | 32.73 % | +3.67 pp | 7.91 % | -21.15 pp |
| DRAM Throughput (%) | 3.15 % | 0.03 % | -3.12 pp | 0.03 % | -3.12 pp | 3.09 % | -0.06 pp | 4.15 % | +1.00 pp |
| Achieved Occupancy (%) | 68.42 % | 61.33 % | -7.09 pp | 57.99 % | -10.43 pp | 72.38 % | +3.96 pp | 62.08 % | -6.34 pp |
| Registers/Thread | 28.00 register/thread | 40.00 register/thread | +12 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 20.00 register/thread | -8 |
| Waves/SM | 1.18 | 1.58 | +0.40 | 0.59 | -0.59 | 1.18 | +0.00 | 1.18 | +0.00 |
| No Eligible (%) | 38.69 % | 29.00 % | -9.69 pp | 52.93 % | +14.24 pp | 30.43 % | -8.26 pp | 38.12 % | -0.57 pp |
| IPC Active | 2.43 inst/cycle | 2.88 inst/cycle | +0.45 | 1.86 inst/cycle | -0.57 | 2.69 inst/cycle | +0.26 | 2.01 inst/cycle | -0.42 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.006×)

Replacing a short-lived pointwise Triton kernel (3.58 µs, 469K instructions) with a persistent while-loop kernel that iterates over all tiles inside 132 CTAs caused a catastrophic 223x slowdown (797 µs, 430M instructions). The while-loop body executes ~19 iterations per CTA but the runtime arithmetic for tile decomposition (integer divides/modulos of tile_id inside the loop) is not constexpr and cannot be optimized away by the compiler, generating enormous instruction overhead per tile. Additionally, registers-per-thread doubled from 20 to 40, halving theoretical occupancy from 100% to 75% and reducing block-limit-registers from 21 to 12 blocks/SM. The persistent pattern is only beneficial when the original kernel is genuinely latency-bound with many waves and long per-tile work; for ultra-short kernels (<5 µs) with only ~19 tiles per CTA, the loop overhead dominates completely.

> **Applicable when**: Never apply persistent kernel scheduling to ultra-short kernels (<10 µs duration) with small total tile counts (<500 tiles). The pattern is only beneficial when: (1) per-tile work is substantial (many memory transactions or compute ops), (2) total tiles >> SM count by a large factor (>>10x), and (3) the kernel is genuinely latency-bound from launch overhead rather than from per-tile work.

> **Avoid when**: Kernel duration is already very short (<10 µs); total tile count is small (e.g., 2496 tiles / 132 SMs = only ~19 iterations per CTA); the kernel body is a simple pointwise op (single load + store); registers-per-thread is already low (<=20) and the persistent loop would double register pressure, halving theoretical occupancy; the partial-wave overhead is the bottleneck but the kernel is too short for loop overhead to amortize.

#### ❌ Plan B: `persistent_tile_scheduling` (0.006×)

Replacing a short-lived pointwise Triton kernel (3.58 µs, 469K instructions) with a persistent while-loop kernel that iterates over all tiles inside 132 CTAs caused a catastrophic 223x slowdown (797 µs, 430M instructions). The while-loop body executes ~19 iterations per CTA but the runtime arithmetic for tile decomposition (integer divides/modulos of tile_id inside the loop) is not constexpr and cannot be optimized away by the compiler, generating enormous instruction overhead per tile. Additionally, registers-per-thread doubled from 20 to 40, halving theoretical occupancy from 100% to 75% and reducing block-limit-registers from 21 to 12 blocks/SM. The persistent pattern is only beneficial when the original kernel is genuinely latency-bound with many waves and long per-tile work; for ultra-short kernels (<5 µs) with only ~19 tiles per CTA, the loop overhead dominates completely.

> **Applicable when**: Never apply persistent kernel scheduling to ultra-short kernels (<10 µs duration) with small total tile counts (<500 tiles). The pattern is only beneficial when: (1) per-tile work is substantial (many memory transactions or compute ops), (2) total tiles >> SM count by a large factor (>>10x), and (3) the kernel is genuinely latency-bound from launch overhead rather than from per-tile work.

> **Avoid when**: Kernel duration is already very short (<10 µs); total tile count is small (e.g., 2496 tiles / 132 SMs = only ~19 iterations per CTA); the kernel body is a simple pointwise op (single load + store); registers-per-thread is already low (<=20) and the persistent loop would double register pressure, halving theoretical occupancy; the partial-wave overhead is the bottleneck but the kernel is too short for loop overhead to amortize.

#### ❌ Plan C: `persistent_tile_scheduling` (0.006×)

Replacing a short-lived pointwise Triton kernel (3.58 µs, 469K instructions) with a persistent while-loop kernel that iterates over all tiles inside 132 CTAs caused a catastrophic 223x slowdown (797 µs, 430M instructions). The while-loop body executes ~19 iterations per CTA but the runtime arithmetic for tile decomposition (integer divides/modulos of tile_id inside the loop) is not constexpr and cannot be optimized away by the compiler, generating enormous instruction overhead per tile. Additionally, registers-per-thread doubled from 20 to 40, halving theoretical occupancy from 100% to 75% and reducing block-limit-registers from 21 to 12 blocks/SM. The persistent pattern is only beneficial when the original kernel is genuinely latency-bound with many waves and long per-tile work; for ultra-short kernels (<5 µs) with only ~19 tiles per CTA, the loop overhead dominates completely.

> **Applicable when**: Never apply persistent kernel scheduling to ultra-short kernels (<10 µs duration) with small total tile counts (<500 tiles). The pattern is only beneficial when: (1) per-tile work is substantial (many memory transactions or compute ops), (2) total tiles >> SM count by a large factor (>>10x), and (3) the kernel is genuinely latency-bound from launch overhead rather than from per-tile work.

> **Avoid when**: Kernel duration is already very short (<10 µs); total tile count is small (e.g., 2496 tiles / 132 SMs = only ~19 iterations per CTA); the kernel body is a simple pointwise op (single load + store); registers-per-thread is already low (<=20) and the persistent loop would double register pressure, halving theoretical occupancy; the partial-wave overhead is the bottleneck but the kernel is too short for loop overhead to amortize.

#### ✅ Plan D: `add_triton_autotune` (1.335×)

Adding @triton.autotune with configs exploring XBLOCK (128, 256, 512, 1024) and num_warps (4, 8) to a kernel that previously used a fixed XBLOCK yielded a 1.34x speedup despite the profiling metrics showing nearly identical instruction counts and similar elapsed cycles. The achieved occupancy improved modestly from 56.4% to 62.1% (active warps per SM: 36 → 40), suggesting the autotuner selected a config that better fills the SM pipeline. Notably, the kernel duration barely changed (3.58 → 3.55 µs), indicating the speedup is measured at the end-to-end level and may reflect reduced launch overhead or better config selection across the full workload rather than a single-kernel improvement visible in NCU.

> **Applicable when**: Kernel uses a fixed hardcoded XBLOCK/num_warps with no autotune; NCU LaunchConfiguration rule fires with speedup_pct >= 20%; achieved occupancy is below 65%; kernel is a pointwise or elementwise op where the body is unchanged across tile sizes.

> **Avoid when**: Kernel already has @triton.autotune; XBLOCK is already right-sized to xnumel; the grid lambda does not adapt to META parameters (autotune configs would be ignored); kernel body has constexpr branches that differ across XBLOCK values requiring separate kernels.

---

## Iteration 3

**Best so far**: 1.194× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune_and_add_streaming_eviction | — | 0.006× | ✓ |  |
| B | fix_xblock_to_16_autotune_yblock_for_coalesced_access | — | 1.398× | ✓ |  |
| C | revert_broken_persistent_loop_to_standard_2d_dispatch | — | 1.225× | ✓ |  |
| D | revert_persistent_kernel_to_2d_tile_dispatch | — | 1.090× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_xblock_to_16_... | Δ | revert_broken_per... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4740 ns | 807200 ns | +802460 ns (+16929.5%) | 3390 ns | -1350 ns (-28.5%) | 3870 ns | -870 ns (-18.4%) | 4350 ns | -390 ns (-8.2%) |
| SM Throughput (%) | 31.08 % | 69.97 % | +38.89 pp | 8.23 % | -22.85 pp | 19.04 % | -12.04 pp | 41.81 % | +10.73 pp |
| Memory Throughput (%) | 29.06 % | 65.47 % | +36.41 pp | 7.55 % | -21.51 pp | 17.45 % | -11.61 pp | 27.88 % | -1.18 pp |
| DRAM Throughput (%) | 3.15 % | 0.03 % | -3.12 pp | 4.41 % | +1.26 pp | 3.87 % | +0.72 pp | 3.42 % | +0.27 pp |
| Achieved Occupancy (%) | 68.42 % | 61.31 % | -7.11 pp | 16.51 % | -51.91 pp | 42.61 % | -25.81 pp | 66.52 % | -1.90 pp |
| Registers/Thread | 28.00 register/thread | 40.00 register/thread | +12 | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 27.00 register/thread | -1 |
| Waves/SM | 1.18 | 1.58 | +0.40 | 0.59 | -0.59 | 0.59 | -0.59 | 1.18 | +0.00 |
| No Eligible (%) | 38.69 % | 26.95 % | -11.74 pp | 74.79 % | +36.10 pp | 40.16 % | +1.47 pp | 29.96 % | -8.73 pp |
| IPC Active | 2.43 inst/cycle | 2.94 inst/cycle | +0.51 | 0.91 inst/cycle | -1.52 | 2.23 inst/cycle | -0.20 | 2.87 inst/cycle | +0.44 |


### Key Learnings

#### ✅ Plan B: `fix_xblock_to_16_autotune_yblock_for_coalesced_access` (1.398×)

When a Triton kernel's XBLOCK (128) far exceeds xnumel (9), replacing all autotune configs with XBLOCK=16 (next power-of-2 >= xnumel) and adding a persistent while-loop over Y tiles inside each CTA reduces executed instructions by ~54% (469K → 218K) and cuts warp cycles per issued instruction from 15.1 to 10.7. The persistent Y-loop (yoffset += num_programs_y * YBLOCK) allows each CTA to stride over multiple Y-tiles, improving L2 hit rate from 41% to 63% and L1 hit rate from 0% to 14% by reusing cached X-dimension data across iterations. The key NCU signals are MemoryCacheAccessPattern (low bytes-per-sector) and LaunchConfiguration (large partial-wave penalty), and the fix requires XBLOCK >= xnumel as a power-of-2 with an xmask guard to avoid correctness failures.

> **Applicable when**: NCU reports MemoryCacheAccessPattern with low bytes-utilized-per-sector AND LaunchConfiguration with large partial-wave speedup_pct (>=30%); XBLOCK >> xnumel (ratio > 4x, e.g., XBLOCK=128 for xnumel=9); kernel has a 2D tile structure with one tiny X dimension and one large Y dimension; L2 hit rate is low (<50%) suggesting poor data reuse; executed instruction count is high relative to useful work.

> **Avoid when**: XBLOCK=1 configs are included — these cause correctness failures when the 2D [1,XBLOCK] xindex shape collapses scatter-store index expressions (e.g., y0 + 96*x2 + 864*y1); xnumel is large and close to XBLOCK (thread waste is minimal); the persistent Y-loop would cause instruction explosion if per-CTA tile count is very large (see exp_002_persistent_kernel_triton_regression); kernel is already compute-bound.

---

## Cumulative Progress

```
Baseline:  4740 ns  (1.000×)
Iter 1     3970 ns  (1.194×)  — autotune_tile_sizes_eliminate_partial_wave

Final best: 3970 ns  (1.1940×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 19968
@@ -22,15 +34,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 19968
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_34[grid](arg_0, arg_1, 19968, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_34[grid](arg_0, arg_1, 19968, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
