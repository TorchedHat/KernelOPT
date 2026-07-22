# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6400 ns — **Best latency**: 4060 ns (**1.5764×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_with_fixed_autotune_callsite`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6400 ns
- **SM Throughput**: 16.68 % | **Memory**: 38.98 % | **DRAM**: 19.28 %
- **Achieved Occupancy**: 55.05 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.324× | 1.905× | ✓ |  |
| B | autotune_tile_sizes_to_fix_uncoalesced_stores_and_occupancy | 1.230× | 1.076× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.377× | 0.985× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_coalescing_and_parallelism | 1.343× | 1.502× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6400 ns | 3360 ns | -3040 ns (-47.5%) | 5950 ns | -450 ns (-7.0%) | 6500 ns | +100 ns (+1.6%) | 4260 ns | -2140 ns (-33.4%) |
| SM Throughput (%) | 16.68 % | 17.75 % | +1.07 pp | 17.55 % | +0.87 pp | 32.21 % | +15.53 pp | 27.77 % | +11.09 pp |
| Memory Throughput (%) | 38.98 % | 20.58 % | -18.40 pp | 29.84 % | -9.14 pp | 31.64 % | -7.34 pp | 21.81 % | -17.17 pp |
| DRAM Throughput (%) | 19.28 % | 4.95 % | -14.33 pp | 10.36 % | -8.92 pp | 9.52 % | -9.76 pp | 3.89 % | -15.39 pp |
| Achieved Occupancy (%) | 55.05 % | 43.92 % | -11.13 pp | 46.05 % | -9.00 pp | 57.47 % | +2.42 pp | 68.11 % | +13.06 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 31.00 register/thread | +5 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.54 % | 53.77 % | -16.77 pp | 65.44 % | -5.10 pp | 40.32 % | -30.22 pp | 47.82 % | -22.72 pp |
| IPC Active | 1.14 inst/cycle | 1.72 inst/cycle | +0.58 | 1.44 inst/cycle | +0.30 | 2.38 inst/cycle | +1.24 | 2.39 inst/cycle | +1.25 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_autotune` (1.905×)

For memory-bound layout permutation kernels suffering from L1TEX stall cycles (CPIStall rule, ~49% of CPI) and low issue slot utilization (~29%), applying tl.max_contiguous and tl.multiple_of hints on the contiguous index dimension enables the Triton compiler to emit 128-bit vectorized loads, dramatically reducing instruction count and L2 sector traffic. Pairing this with @triton.autotune across XBLOCK/YBLOCK/num_warps/num_stages configurations allows the runtime to select the tile shape that best hides remaining memory latency, improving eligible warps per scheduler from 0.99 to 2.07 and issue slot utilization from 29% to 46%. The combined effect reduced kernel duration by ~47% (6.4 µs → 3.36 µs) and L2 sector traffic by ~71% (257K → 75K sectors).

> **Applicable when**: Triton pointwise/permutation kernel with: (1) CPIStall rule triggered with L1TEX stall >40% of CPI; (2) UncoalescedGlobalAccess rule with excessive sectors >10% of total; (3) IssueSlotUtilization low (<35% issue slots busy); (4) eligible warps per scheduler <1.5; (5) load dimension is contiguous in memory (stride-1 access pattern); (6) fixed tile size with no autotune

> **Avoid when**: Load dimension is not contiguous (strided loads won't benefit from vectorization hints); kernel is compute-bound rather than memory-latency-bound; grid is already large enough to saturate SMs (waves_per_sm >> 1); XBLOCK is already at maximum practical size; the kernel uses indirect/gather memory access patterns where contiguity hints are invalid

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_coalesced_stores_via_yblock_expansion | 1.325× | 0.299× | ✓ |  |
| B | vectorized_loads_with_fixed_autotune_callsite | 1.381× | 1.576× | ✓ | **✓ best** |
| C | vectorized_loads | 1.280× | 1.123× | ✓ |  |
| D | vectorized_loads | 1.374× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads | Δ | vectorized_loads | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6400 ns | 21410 ns | +15010 ns (+234.5%) | 4060 ns | -2340 ns (-36.6%) | 5700 ns | -700 ns (-10.9%) | 6460 ns | +60 ns (+0.9%) |
| SM Throughput (%) | 16.68 % | 37.38 % | +20.70 pp | 27.08 % | +10.40 pp | 18.70 % | +2.02 pp | 31.78 % | +15.10 pp |
| Memory Throughput (%) | 38.98 % | 51.67 % | +12.69 pp | 21.49 % | -17.49 pp | 30.98 % | -8.00 pp | 31.16 % | -7.82 pp |
| DRAM Throughput (%) | 19.28 % | 5.75 % | -13.53 pp | 4.05 % | -15.23 pp | 10.79 % | -8.49 pp | 9.56 % | -9.72 pp |
| Achieved Occupancy (%) | 55.05 % | 65.88 % | +10.83 pp | 68.59 % | +13.54 pp | 46.41 % | -8.64 pp | 57.76 % | +2.71 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 31.00 register/thread | +5 |
| Waves/SM | 0.74 | 1.98 | +1.24 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.54 % | 55.69 % | -14.85 pp | 37.25 % | -33.29 pp | 61.84 % | -8.70 pp | 39.69 % | -30.85 pp |
| IPC Active | 1.14 inst/cycle | 1.77 inst/cycle | +0.63 | 2.39 inst/cycle | +1.25 | 1.45 inst/cycle | +0.31 | 2.38 inst/cycle | +1.24 |

---

## Iteration 3

**Best so far**: 1.5764× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.5764× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_partial_wave_via_persistent_kernel | 0.314× | 0.266× | ✓ |  |
| C | small_xblock_large_yblock_to_coalesce_strided_stores | 1.335× | 1.527× | ✓ |  |
| D | add_triton_autotune | 1.208× | 1.063× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | small_xblock_larg... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6400 ns | 24100 ns | +17700 ns (+276.6%) | 4190 ns | -2210 ns (-34.5%) | 6020 ns | -380 ns (-5.9%) |
| SM Throughput (%) | 16.68 % | 18.06 % | +1.38 pp | 11.02 % | -5.66 pp | 16.88 % | +0.20 pp |
| Memory Throughput (%) | 38.98 % | 78.97 % | +39.99 pp | 21.01 % | -17.97 pp | 41.94 % | +2.96 pp |
| DRAM Throughput (%) | 19.28 % | 5.13 % | -14.15 pp | 3.72 % | -15.56 pp | 20.55 % | +1.27 pp |
| Achieved Occupancy (%) | 55.05 % | 80.24 % | +25.19 pp | 20.79 % | -34.26 pp | 56.07 % | +1.02 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.54 % | 78.01 % | +7.47 pp | 68.99 % | -1.55 pp | 70.98 % | +0.44 pp |
| IPC Active | 1.14 inst/cycle | 0.89 inst/cycle | -0.25 | 1.09 inst/cycle | -0.05 | 1.08 inst/cycle | -0.06 |


### Key Learnings

#### ❌ Plan B: `eliminate_partial_wave_via_persistent_kernel` (0.266×)

Converting a 2D-grid pointwise kernel to a 1D persistent kernel with a software tile loop (tile_id += num_programs) caused a 3.76x slowdown despite the NCU LaunchConfiguration rule suggesting 50% savings from partial-wave elimination. The flat-index decomposition (x2 = idx % xnumel, y3 = idx // xnumel) introduced expensive integer division and modulo operations per element, increasing executed instructions by 4.07x (868K → 3.53M) and register pressure from 18 to 32 registers/thread. The resulting L1TEX scoreboard stalls dominated execution, raising warp cycles per issued instruction from 17.6 to 57.7 and collapsing eligible warps/scheduler from 3.96 to 0.63, far outweighing any benefit from eliminating the partial wave.

> **Applicable when**: Never: this pattern is an anti-pattern. Do not apply persistent kernel with flat-index decomposition to replace 2D-grid pointwise kernels when the 2D indexing is already natural and cheap.

> **Avoid when**: Any pointwise or layout-permutation kernel where the natural 2D index decomposition (program_id(0), program_id(1)) is already available. The flat-index decomposition (idx % stride, idx // stride) adds O(N) integer divisions that cost far more than the partial-wave tail penalty flagged by NCU LaunchConfiguration. Specifically avoid when: (1) NCU LaunchConfiguration speedup_pct <= 50% and kernel duration is already short (<10 us); (2) the kernel has low register count (<24) that would increase with the added ALU; (3) the existing 2D grid already provides sufficient occupancy (achieved occupancy >60%).

---

## Cumulative Progress

```
Baseline:  6400 ns  (1.000×)
Iter 2     4060 ns  (1.576×)  — vectorized_loads_with_fixed_autotune_callsite

Final best: 4060 ns  (1.5764×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=2, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
