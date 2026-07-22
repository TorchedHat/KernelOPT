# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6300 ns — **Best latency**: 3360 ns (**1.8750×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_via_contiguity_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6300 ns
- **SM Throughput**: 16.68 % | **Memory**: 39.51 % | **DRAM**: 19.63 %
- **Achieved Occupancy**: 54.75 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.266× | 1.552× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.413× | 1.875× | ✓ | **✓ best** |
| C | memory_coalescing_swap_tile_dimensions | 1.065× | 1.000× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 0.995× | 0.461× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6300 ns | 4060 ns | -2240 ns (-35.6%) | 3360 ns | -2940 ns (-46.7%) | 6300 ns | +0 ns (+0.0%) | 13660 ns | +7360 ns (+116.8%) |
| SM Throughput (%) | 16.68 % | 26.45 % | +9.77 pp | 17.82 % | +1.14 pp | 14.42 % | -2.26 pp | 49.90 % | +33.22 pp |
| Memory Throughput (%) | 39.51 % | 21.03 % | -18.48 pp | 20.43 % | -19.08 pp | 26.33 % | -13.18 pp | 61.58 % | +22.07 pp |
| DRAM Throughput (%) | 19.63 % | 4.06 % | -15.57 pp | 4.95 % | -14.68 pp | 4.89 % | -14.74 pp | 9.00 % | -10.63 pp |
| Achieved Occupancy (%) | 54.75 % | 67.61 % | +12.86 pp | 39.61 % | -15.14 pp | 43.97 % | -10.78 pp | 63.86 % | +9.11 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 24.00 register/thread | -2 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.98 | +1.24 |
| No Eligible (%) | 70.81 % | 37.37 % | -33.44 pp | 55.46 % | -15.35 pp | 67.88 % | -2.93 pp | 44.92 % | -25.89 pp |
| IPC Active | 1.14 inst/cycle | 2.38 inst/cycle | +1.24 | 1.40 inst/cycle | +0.26 | 1.20 inst/cycle | +0.06 | 2.21 inst/cycle | +1.07 |


### Key Learnings

#### ❌ Plan D: `autotune_tile_sizes_for_occupancy_and_coalescing` (0.461×)

Adding @triton.autotune with larger XBLOCK/YBLOCK configs caused a 2.17x slowdown (6.3 µs → 13.66 µs) despite improving waves/SM (0.74→1.98), achieved occupancy (54.7%→63.9%), and issue slot utilization (29.1%→55.4%). The regression was caused by the autotune selecting a config with large shared memory per block (32.77 KB vs 4.1 KB), which reduced theoretical occupancy from 100% to 75% and triggered a new 'TheoreticalOccupancy limited by shared memory' rule. Additionally, the selected config doubled the instruction count (771K→4.3M executed instructions) and introduced MIO pipeline stalls (6.1 cycles, 32.9% of stall budget) from heavy shared memory usage, overwhelming the gains from better warp eligibility. When the kernel is already small (sub-10 µs, <1 wave), autotune overhead and shared-memory-heavy configs can dominate, making the net result worse.

> **Applicable when**: Kernel has <1 wave/SM, >60% no-eligible-warp cycles, and low achieved occupancy (<60%) with theoretical occupancy near 100% — AND the kernel uses minimal shared memory in its baseline configuration.

> **Avoid when**: The kernel is already very short (<10 µs) and the baseline uses minimal shared memory (<8 KB/block). Adding autotune configs that allocate large shared memory per block will reduce theoretical occupancy and introduce MIO pipeline stalls, causing net regression even if warp eligibility improves. Also avoid when the autotune search space includes configs that more than double the instruction count relative to the baseline.

---

## Iteration 2

**Best so far**: 1.875× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguity_hints | 1.385× | 1.875× | ✓ |  |
| B | eliminate_partial_wave_via_grid_aligned_tile_sizes_with_num_stages | 1.265× | 0.903× | ✓ |  |
| C | vectorized_loads_via_contiguity_hints_and_fix_autotune_grid | 1.391× | 1.458× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | eliminate_partial... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6300 ns | 3360 ns | -2940 ns (-46.7%) | 6980 ns | +680 ns (+10.8%) | 4320 ns | -1980 ns (-31.4%) |
| SM Throughput (%) | 16.68 % | 17.79 % | +1.11 pp | 34.71 % | +18.03 pp | 27.79 % | +11.11 pp |
| Memory Throughput (%) | 39.51 % | 20.31 % | -19.20 pp | 45.58 % | +6.07 pp | 22.05 % | -17.46 pp |
| DRAM Throughput (%) | 19.63 % | 4.94 % | -14.69 pp | 2.36 % | -17.27 pp | 3.84 % | -15.79 pp |
| Achieved Occupancy (%) | 54.75 % | 44.34 % | -10.41 pp | 77.57 % | +22.82 pp | 69.57 % | +14.82 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 32.00 register/thread | +6 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.81 % | 56.58 % | -14.23 pp | 41.04 % | -29.77 pp | 37.25 % | -33.56 pp |
| IPC Active | 1.14 inst/cycle | 1.69 inst/cycle | +0.55 | 2.33 inst/cycle | +1.19 | 2.41 inst/cycle | +1.27 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_contiguity_hints` (1.875×)

Wrapping the contiguous load index with tl.max_contiguous and tl.multiple_of hints in Triton signals to the compiler that the access is aligned and contiguous, enabling wider (128-bit) memory transactions. In this case, the hints improved Executed IPC Active from 1.40 to 1.69 and Issue Slots Busy from 37.2% to 44.7%, and reduced Warp Cycles Per Executed Instruction from 21.0 to 16.5, indicating better instruction throughput per active cycle. However, the wall-clock duration remained unchanged (3.36 µs) because the kernel was already bottlenecked by an uncoalesced strided store (stride-3 in the output), which the load hints cannot address — the speedup reported (1.875x) reflects a broader optimization context rather than this hint change alone.

> **Applicable when**: Triton kernel with a contiguous load dimension (stride-1 in the XBLOCK direction) where xoffset = program_id(0) * XBLOCK is provably a multiple of XBLOCK; UncoalescedGlobalAccess rule triggered; IssueSlotUtilization shows >50% no-eligible-warp cycles; load index is a simple linear expression in tl.arange(0, XBLOCK).

> **Avoid when**: The store access pattern is the dominant bottleneck (e.g., stride-3 or other non-unit-stride output layout) — load hints alone cannot fix store-side uncoalescing and wall-clock improvement will be negligible. Do not apply tl.multiple_of if xoffset is not guaranteed to be a multiple of XBLOCK for all autotune configs. Do not apply to the store index if it has non-unit stride.

---

## Iteration 3

**Best so far**: 1.875× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | streaming_load_eviction_policy_to_reduce_l1_contention | 1.346× | 1.493× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_load_ev... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6300 ns | 4220 ns | -2080 ns (-33.0%) |
| SM Throughput (%) | 16.68 % | 24.96 % | +8.28 pp |
| Memory Throughput (%) | 39.51 % | 19.82 % | -19.69 pp |
| DRAM Throughput (%) | 19.63 % | 3.92 % | -15.71 pp |
| Achieved Occupancy (%) | 54.75 % | 68.17 % | +13.42 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.81 % | 38.19 % | -32.62 pp |
| IPC Active | 1.14 inst/cycle | 2.39 inst/cycle | +1.25 |

---

## Cumulative Progress

```
Baseline:  6300 ns  (1.000×)
Iter 1     3360 ns  (1.875×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3360 ns  (1.8750×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +31,13 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(
+        tl.multiple_of(
+            xoffset + tl.arange(0, XBLOCK)[None, :],
+            [1, XBLOCK]
+        ),
+        [1, XBLOCK]
+    )
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +46,13 @@

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
