# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3360 ns (**1.7708×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`stride3_aligned_yblock_autotune_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.62 % | **Memory**: 41.22 % | **DRAM**: 20.67 %
- **Achieved Occupancy**: 54.76 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.353× | 1.617× | ✓ | **✓ best** |
| B | fix_uncoalesced_stores_via_autotune_tile_reshape | 1.264× | 1.420× | ✓ |  |
| C | memory_coalescing_via_tile_dimension_swap_and_autotune | 1.317× | 1.291× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.144× | 1.074× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3680 ns | -2270 ns (-38.2%) | 4190 ns | -1760 ns (-29.6%) | 4610 ns | -1340 ns (-22.5%) | 5540 ns | -410 ns (-6.9%) |
| SM Throughput (%) | 16.62 % | 25.97 % | +9.35 pp | 17.17 % | +0.55 pp | 25.06 % | +8.44 pp | 34.49 % | +17.87 pp |
| Memory Throughput (%) | 41.22 % | 18.57 % | -22.65 pp | 33.16 % | -8.06 pp | 22.05 % | -19.17 pp | 19.58 % | -21.64 pp |
| DRAM Throughput (%) | 20.67 % | 4.51 % | -16.16 pp | 15.78 % | -4.89 pp | 6.68 % | -13.99 pp | 11.15 % | -9.52 pp |
| Achieved Occupancy (%) | 54.76 % | 66.81 % | +12.05 pp | 54.34 % | -0.42 pp | 53.23 % | -1.53 pp | 79.27 % | +24.51 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 21.00 register/thread | -5 | 28.00 register/thread | +2 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 2.97 | +2.23 |
| No Eligible (%) | 70.74 % | 42.21 % | -28.53 pp | 64.75 % | -5.99 pp | 39.33 % | -31.41 pp | 45.58 % | -25.16 pp |
| IPC Active | 1.14 inst/cycle | 2.24 inst/cycle | +1.10 | 1.33 inst/cycle | +0.19 | 2.38 inst/cycle | +1.24 | 2.11 inst/cycle | +0.97 |

---

## Iteration 2

**Best so far**: 1.6168× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.238× | 1.368× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.314× | 1.291× | ✓ |  |
| D | vectorized_loads | 1.391× | 1.420× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 4350 ns | -1600 ns (-26.9%) | 4610 ns | -1340 ns (-22.5%) | 4190 ns | -1760 ns (-29.6%) |
| SM Throughput (%) | 16.62 % | 17.14 % | +0.52 pp | 24.81 % | +8.19 pp | 16.55 % | -0.07 pp |
| Memory Throughput (%) | 41.22 % | 31.87 % | -9.35 pp | 21.79 % | -19.43 pp | 27.29 % | -13.93 pp |
| DRAM Throughput (%) | 20.67 % | 15.15 % | -5.52 pp | 6.71 % | -13.96 pp | 14.75 % | -5.92 pp |
| Achieved Occupancy (%) | 54.76 % | 54.60 % | -0.16 pp | 53.05 % | -1.71 pp | 57.48 % | +2.72 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 28.00 register/thread | +2 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.74 % | 64.81 % | -5.93 pp | 39.13 % | -31.61 pp | 63.67 % | -7.07 pp |
| IPC Active | 1.14 inst/cycle | 1.32 inst/cycle | +0.18 | 2.36 inst/cycle | +1.22 | 1.37 inst/cycle | +0.23 |

---

## Iteration 3

**Best so far**: 1.6168× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_store_via_arange_axis_swap_with_aligned_autotune | 1.244× | 0.925× | ✓ |  |
| C | software_pipelining_via_num_stages_in_autotune_configs | 1.342× | 1.346× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_store_v... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 6430 ns | +480 ns (+8.1%) | 4420 ns | -1530 ns (-25.7%) |
| SM Throughput (%) | 16.62 % | 32.51 % | +15.89 pp | 25.21 % | +8.59 pp |
| Memory Throughput (%) | 41.22 % | 31.88 % | -9.34 pp | 22.19 % | -19.03 pp |
| DRAM Throughput (%) | 20.67 % | 9.58 % | -11.09 pp | 7.01 % | -13.66 pp |
| Achieved Occupancy (%) | 54.76 % | 57.63 % | +2.87 pp | 54.58 % | -0.18 pp |
| Registers/Thread | 26.00 register/thread | 31.00 register/thread | +5 | 28.00 register/thread | +2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.74 % | 39.49 % | -31.25 pp | 39.47 % | -31.27 pp |
| IPC Active | 1.14 inst/cycle | 2.39 inst/cycle | +1.25 | 2.13 inst/cycle | +0.99 |

---

## Iteration 4

**Best so far**: 1.6168× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | yblock_period3_alignment_with_lambda_grid | 1.348× | 0.250× | ✓ |  |
| B | stride3_aligned_yblock_autotune_for_occupancy_and_coalescing | 1.421× | 1.771× | ✓ | **✓ best** |
| C | coalesced_store_via_yblock_warpwide_axis_with_expanded_autotune | 1.205× | 1.328× | ✓ |  |
| D | persistent_flat_grid_to_fill_sm_waves | 0.453× | 0.251× | ✓ |  |


### Metric Comparison

| Metric | Baseline | yblock_period3_al... | Δ | stride3_aligned_y... | Δ | coalesced_store_v... | Δ | persistent_flat_g... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 23840 ns | +17890 ns (+300.7%) | 3360 ns | -2590 ns (-43.5%) | 4480 ns | -1470 ns (-24.7%) | 23680 ns | +17730 ns (+298.0%) |
| SM Throughput (%) | 16.62 % | 53.41 % | +36.79 pp | 14.78 % | -1.84 pp | 24.91 % | +8.29 pp | 26.89 % | +10.27 pp |
| Memory Throughput (%) | 41.22 % | 56.42 % | +15.20 pp | 20.58 % | -20.64 pp | 23.19 % | -18.03 pp | 80.45 % | +39.23 pp |
| DRAM Throughput (%) | 20.67 % | 5.15 % | -15.52 pp | 4.95 % | -15.72 pp | 7.35 % | -13.32 pp | 5.31 % | -15.36 pp |
| Achieved Occupancy (%) | 54.76 % | 67.35 % | +12.59 pp | 45.23 % | -9.53 pp | 70.09 % | +15.33 pp | 86.81 % | +32.05 pp |
| Registers/Thread | 26.00 register/thread | 40.00 register/thread | +14 | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 3.96 | +3.22 | 0.74 | +0.00 | 1.48 | +0.74 | 2.97 | +2.23 |
| No Eligible (%) | 70.74 % | 38.83 % | -31.91 pp | 62.70 % | -8.04 pp | 49.43 % | -21.31 pp | 69.59 % | -1.15 pp |
| IPC Active | 1.14 inst/cycle | 2.44 inst/cycle | +1.30 | 1.47 inst/cycle | +0.33 | 1.89 inst/cycle | +0.75 | 1.20 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan D: `persistent_flat_grid_to_fill_sm_waves` (0.254×)

Converting a 2D grid kernel to a persistent flat-grid while-loop (1D grid, each CTA iterates over multiple tiles) caused a 3.9x slowdown (3.36 µs → 23.42 µs). Although waves/SM rose from 0.74 to 2.97 and achieved occupancy improved from 45.2% to 87.1%, the L1/TEX hit rate collapsed from 0% (already poor) to 1.54% and L2 throughput saturated at 84.4%, while executed instructions ballooned 13x (367K → 4.8M) and warp cycles per issued instruction tripled from 18.7 to 45.7. The persistent loop forces each CTA to access a wide, non-contiguous range of input addresses across tile iterations, destroying any residual L1 locality and creating severe L1TEX scoreboard stalls (83.9% of stall cycles). This pattern is counterproductive for strided/uncoalesced layout-permutation kernels where the per-CTA working set across iterations far exceeds L1 capacity.

> **Applicable when**: Never apply persistent flat-grid loops to kernels with strided/uncoalesced memory access patterns where the per-CTA working set across multiple tile iterations exceeds L1 cache capacity. The pattern is only beneficial when tiles share significant data reuse (e.g., matmul with shared memory staging).

> **Avoid when**: Kernel has uncoalesced global accesses (UncoalescedGlobalAccess rule triggered, >10% excess sectors); kernel is a pure layout permutation or copy with no data reuse between tiles; L1/TEX hit rate is already low (<10%) in the baseline — the persistent loop will make it worse; kernel instruction count is already low (<1M instructions) indicating the baseline is not compute-heavy.

---

## Iteration 5

**Best so far**: 1.7708× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_and_store_address_hints | 1.296× | 0.245× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_tile_exploration | 0.451× | 0.254× | ✓ |  |
| C | yblock_period3_grid_lambda_fix_with_aligned_autotune | 0.454× | 0.254× | ✓ |  |
| D | fix_autotune_grid_lambda_with_register_reducing_configs | 1.419× | 1.877× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_autotune_grid... | Δ | yblock_period3_gr... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 24260 ns | +18310 ns (+307.7%) | 23420 ns | +17470 ns (+293.6%) | 23420 ns | +17470 ns (+293.6%) | 3170 ns | -2780 ns (-46.7%) |
| SM Throughput (%) | 16.62 % | 53.65 % | +37.03 pp | 27.22 % | +10.60 pp | 26.83 % | +10.21 pp | 14.97 % | -1.65 pp |
| Memory Throughput (%) | 41.22 % | 56.11 % | +14.89 pp | 81.38 % | +40.16 pp | 81.39 % | +40.17 pp | 21.76 % | -19.46 pp |
| DRAM Throughput (%) | 20.67 % | 5.07 % | -15.60 pp | 5.40 % | -15.27 pp | 5.39 % | -15.28 pp | 5.23 % | -15.44 pp |
| Achieved Occupancy (%) | 54.76 % | 67.56 % | +12.80 pp | 87.07 % | +32.31 pp | 86.60 % | +31.84 pp | 47.62 % | -7.14 pp |
| Registers/Thread | 26.00 register/thread | 40.00 register/thread | +14 | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 3.96 | +3.22 | 2.97 | +2.23 | 2.97 | +2.23 | 0.74 | +0.00 |
| No Eligible (%) | 70.74 % | 38.68 % | -32.06 pp | 69.54 % | -1.20 pp | 69.38 % | -1.36 pp | 61.26 % | -9.48 pp |
| IPC Active | 1.14 inst/cycle | 2.45 inst/cycle | +1.31 | 1.21 inst/cycle | +0.07 | 1.21 inst/cycle | +0.07 | 1.36 inst/cycle | +0.22 |


### Key Learnings

#### ❌ Plan A: `streaming_eviction_policy_and_store_address_hints` (0.245×)

Applying evict_first eviction policy to loads/stores and adding tl.max_contiguous/tl.multiple_of store address hints caused a 4x slowdown (3.36 µs → 24.26 µs). The regression was not caused by the eviction policy change itself but by the autotune config reshuffle that accompanied it: the new configs biased toward large block sizes (XBLOCK=512, YBLOCK=32), which inflated register pressure from 16 to 40 registers/thread, reduced theoretical occupancy from 100% to 75%, and exploded executed instructions 27x (366K → 10.1M). The tl.max_contiguous hint on a 2D index (x2_hint = tl.max_contiguous(tl.multiple_of(x2, [XBLOCK, 1]), [XBLOCK, 1])) is syntactically incorrect for a scalar/1D index and likely caused the compiler to emit a much larger, unrolled code path. The eviction policy change alone (evict_last → evict_first) is a low-risk hint that does not explain the 27x instruction inflation; the root cause is the autotune config list change combined with the malformed hint.

> **Applicable when**: Never apply this pattern as implemented. The eviction policy change alone (evict_last → evict_first) may be safe for streaming copy kernels, but must not be bundled with autotune config reshuffles or malformed tl.max_contiguous hints.

> **Avoid when**: Do not use tl.multiple_of(x2, [XBLOCK, 1]) or tl.max_contiguous(x2, [XBLOCK, 1]) with list arguments when x2 is a 1D or scalar index — these functions expect scalar integer alignment values, not lists. Do not reshuffle autotune configs toward larger XBLOCK/YBLOCK values without verifying register pressure stays below 32 registers/thread (to maintain 100% theoretical occupancy on H100/H200).

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3680 ns  (1.617×)  — vectorized_loads_with_autotune
Iter 4     3360 ns  (1.771×)  — stride3_aligned_yblock_autotune_for_occupancy_and_coalescing

Final best: 3360 ns  (1.7708×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 1,  'XBLOCK': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'YBLOCK': 1,  'XBLOCK': 512},  num_warps=8,  num_stages=4),
+        triton.Config({'YBLOCK': 1,  'XBLOCK': 1024}, num_warps=16, num_stages=4),
+        triton.Config({'YBLOCK': 2,  'XBLOCK': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'YBLOCK': 2,  'XBLOCK': 512},  num_warps=8,  num_stages=4),
+        triton.Config({'YBLOCK': 2,  'XBLOCK': 1024}, num_warps=16, num_stages=4),
+        triton.Config({'YBLOCK': 4,  'XBLOCK': 256},  num_warps=8,  num_stages=3),
+        triton.Config({'YBLOCK': 4,  'XBLOCK': 512},  num_warps=8,  num_stages=4),
+        triton.Config({'YBLOCK': 4,  'XBLOCK': 1024}, num_warps=16, num_stages=4),
+        triton.Config({'YBLOCK': 8,  'XBLOCK': 128},  num_warps=4,  num_stages=3),
+        triton.Config({'YBLOCK': 8,  'XBLOCK': 256},  num_warps=8,  num_stages=3),
+        triton.Config({'YBLOCK': 8,  'XBLOCK': 512},  num_warps=16, num_stages=4),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 64},   num_warps=4,  num_stages=3),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 128},  num_warps=4,  num_stages=3),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 256},  num_warps=8,  num_stages=3),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 512},  num_warps=8,  num_stages=4),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 1024}, num_warps=16, num_stages=4),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 64},   num_warps=4,  num_stages=3),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 128},  num_warps=8,  num_stages=3),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 256},  num_warps=8,  num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
     xnumel = 50176
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[:, None]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +48,13 @@

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

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
