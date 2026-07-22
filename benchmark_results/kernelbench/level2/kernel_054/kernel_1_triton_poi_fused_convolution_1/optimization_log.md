# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285470 ns — **Best latency**: 263140 ns (**1.0849×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints_on_coalesced_axis`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_054/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285470 ns
- **SM Throughput**: 19.61 % | **Memory**: 78.86 % | **DRAM**: 72.26 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_tile_dimensions | 2.119× | 1.089× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285470 ns | 262110 ns | -23360 ns (-8.2%) |
| SM Throughput (%) | 19.61 % | 47.25 % | +27.64 pp |
| Memory Throughput (%) | 78.86 % | 90.95 % | +12.09 pp |
| DRAM Throughput (%) | 72.26 % | 40.27 % | -31.99 pp |
| Achieved Occupancy (%) | 94.05 % | 58.05 % | -36.00 pp |
| Registers/Thread | 25.00 register/thread | 48.00 register/thread | +23 |
| Waves/SM | 59.88 | 95.81 | +35.93 |
| No Eligible (%) | 80.34 % | 52.09 % | -28.25 pp |
| IPC Active | 0.78 inst/cycle | 1.92 inst/cycle | +1.14 |

---

## Iteration 2

**Best so far**: 1.0891× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints_on_coalesced_axis | 3.518× | 1.085× | ✓ | **✓ best** |
| C | coalesce_stores_by_making_spatial_axis_warp_fast | 1.770× | 1.559× | ✓ |  |
| D | fix_autotune_grid_and_expand_configs_for_store_coalescing | 3.507× | 1.089× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | coalesce_stores_b... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285470 ns | 263140 ns | -22330 ns (-7.8%) | 183140 ns | -102330 ns (-35.8%) | 262050 ns | -23420 ns (-8.2%) |
| SM Throughput (%) | 19.61 % | 47.33 % | +27.72 pp | 27.47 % | +7.86 pp | 47.26 % | +27.65 pp |
| Memory Throughput (%) | 78.86 % | 91.10 % | +12.24 pp | 90.29 % | +11.43 pp | 90.95 % | +12.09 pp |
| DRAM Throughput (%) | 72.26 % | 40.11 % | -32.15 pp | 55.56 % | -16.70 pp | 40.30 % | -31.96 pp |
| Achieved Occupancy (%) | 94.05 % | 57.95 % | -36.10 pp | 92.86 % | -1.19 pp | 58.06 % | -35.99 pp |
| Registers/Thread | 25.00 register/thread | 48.00 register/thread | +23 | 23.00 register/thread | -2 | 48.00 register/thread | +23 |
| Waves/SM | 59.88 | 95.81 | +35.93 | 59.88 | +0.00 | 95.81 | +35.93 |
| No Eligible (%) | 80.34 % | 52.14 % | -28.20 pp | 72.54 % | -7.80 pp | 52.09 % | -28.25 pp |
| IPC Active | 0.78 inst/cycle | 1.91 inst/cycle | +1.13 | 1.11 inst/cycle | +0.33 | 1.91 inst/cycle | +1.13 |


### Key Learnings

#### ✅ Plan C: `coalesce_stores_by_making_spatial_axis_warp_fast` (1.567×)

In a Triton transpose kernel where loads are already coalesced but stores remain uncoalesced (NCU MemoryCacheAccessPattern reports <30/32 bytes utilized per store sector), swapping the warp-fast (X) dimension to cover the stride-1 axis of the output tensor coalesces stores and dramatically reduces register pressure and shared memory usage. In this case, reassigning XBLOCK to the spatial HW axis (stride-1 in output) dropped registers/thread from 48 to 23 and dynamic shared memory from 16 KB to 2 KB/block, raising theoretical occupancy from 62.5% to 100% and achieved occupancy from 58% to 93%, which more than compensated for the increased warp stall cycles caused by re-introducing strided loads. The net effect was a 1.56x speedup driven primarily by the occupancy and register pressure improvement rather than pure coalescing; the iteration-3 autotune config refinement produced only marginal additional gains (~0.5%) on top of the already-optimized kernel.

> **Applicable when**: Triton transpose/copy kernel where loads are already coalesced but stores are uncoalesced (NCU MemoryCacheAccessPattern fires on stores, <30/32 bytes utilized per store sector, UncoalescedGlobalAccess reports significant excessive sectors); register pressure is high (>40 registers/thread) limiting occupancy (<70% theoretical); the output tensor has a stride-1 axis that is NOT currently the warp-fast dimension.

> **Avoid when**: Both loads and stores are already coalesced; the kernel has non-trivial arithmetic where reordering dimensions changes computation semantics; the stride-1 output axis is already the warp-fast dimension; register pressure is already low and occupancy is already near 100% (the warp stall increase from strided loads would dominate without an occupancy gain to compensate).

---

## Iteration 3

**Best so far**: 1.0891× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_channel_axis_contiguous_hints | 3.159× | 1.569× | ✓ |  |
| C | coalesce_stores_swap_xblock_to_spatial_hw_axis | 1.994× | 1.567× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | coalesce_stores_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285470 ns | 181980 ns | -103490 ns (-36.3%) | 182180 ns | -103290 ns (-36.2%) |
| SM Throughput (%) | 19.61 % | 27.08 % | +7.47 pp | 27.44 % | +7.83 pp |
| Memory Throughput (%) | 78.86 % | 90.77 % | +11.91 pp | 90.65 % | +11.79 pp |
| DRAM Throughput (%) | 72.26 % | 55.92 % | -16.34 pp | 55.85 % | -16.41 pp |
| Achieved Occupancy (%) | 94.05 % | 92.90 % | -1.15 pp | 92.85 % | -1.20 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 | 23.00 register/thread | -2 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.34 % | 72.44 % | -7.90 pp | 72.41 % | -7.93 pp |
| IPC Active | 0.78 inst/cycle | 1.10 inst/cycle | +0.32 | 1.12 inst/cycle | +0.34 |

---

## Iteration 4

**Best so far**: 1.0891× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_with_contiguous_hints_and_increased_pipeline_stages | 3.156× | 1.566× | ✓ |  |
| B | shared_memory_tiled_transpose_coalesce_loads_and_stores | 1.757× | 0.893× | ✓ |  |
| C | vectorized_loads_contiguous_hints_on_channel_axis | 1.998× | 1.570× | ✓ |  |
| D | cache_hint_eviction_policy_and_expanded_num_stages_for_strided_load_latency_hiding | 1.994× | 1.562× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | shared_memory_til... | Δ | vectorized_loads_... | Δ | cache_hint_evicti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285470 ns | 182270 ns | -103200 ns (-36.2%) | 319520 ns | +34050 ns (+11.9%) | 181820 ns | -103650 ns (-36.3%) | 182750 ns | -102720 ns (-36.0%) |
| SM Throughput (%) | 19.61 % | 27.32 % | +7.71 pp | 32.31 % | +12.70 pp | 27.17 % | +7.56 pp | 27.14 % | +7.53 pp |
| Memory Throughput (%) | 78.86 % | 90.64 % | +11.78 pp | 77.42 % | -1.44 pp | 90.86 % | +12.00 pp | 90.37 % | +11.51 pp |
| DRAM Throughput (%) | 72.26 % | 55.81 % | -16.45 pp | 34.27 % | -37.99 pp | 55.96 % | -16.30 pp | 55.66 % | -16.60 pp |
| Achieved Occupancy (%) | 94.05 % | 92.84 % | -1.21 pp | 93.53 % | -0.52 pp | 92.90 % | -1.15 pp | 92.86 % | -1.19 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 | 28.00 register/thread | +3 | 23.00 register/thread | -2 | 23.00 register/thread | -2 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.34 % | 72.30 % | -8.04 pp | 70.62 % | -9.72 pp | 72.28 % | -8.06 pp | 72.21 % | -8.13 pp |
| IPC Active | 0.78 inst/cycle | 1.11 inst/cycle | +0.33 | 1.17 inst/cycle | +0.39 | 1.10 inst/cycle | +0.32 | 1.11 inst/cycle | +0.33 |


### Key Learnings

#### ✅ Plan C: `vectorized_loads_contiguous_hints_on_channel_axis` (1.570×)

Adding tl.multiple_of and tl.max_contiguous hints to the xindex (spatial/warp-fast) axis in addition to the yindex (NC/slow) axis, and expanding the autotune config space with larger YBLOCK values (16, 32) and additional XBLOCK variants, produced a 1.57x speedup. However, NCU metrics show virtually no change between before and after (duration 181.98 us vs 181.82 us, L2 throughput 94.87% vs 94.90%, warp cycles 53.2 vs 53.4, excessive sectors 8.3M unchanged), confirming the NCU-measured kernel is essentially identical. The speedup is attributable to the expanded autotune config space selecting a better tile configuration at the host/dispatch level rather than the contiguity hints changing PTX instruction emission. When the kernel is already L2-saturated (>90% Mem Busy, ~95% L2 throughput) with persistent uncoalesced loads (16/32 bytes per sector), adding compiler metadata hints to either dimension does not resolve the structural uncoalescing problem — the gains come from autotune exploration.

> **Applicable when**: Triton transpose/copy kernel where the autotune config space has not yet been fully explored; adding contiguity hints to both X and Y index expressions and expanding configs with new XBLOCK/YBLOCK combinations may allow autotune to find a marginally better tile shape.

> **Avoid when**: NCU metrics show no change in bytes-utilized-per-sector or excessive sector count after applying hints (as seen here — 16/32 bytes and 8.3M excessive sectors unchanged), indicating the structural uncoalescing is in the warp-fast (X) dimension and hints on either dimension cannot fix it; the kernel is already L2-saturated with no headroom from vectorization hints alone; the autotune config space has already been exhaustively explored in prior iterations.

---

## Cumulative Progress

```
Baseline:  285470 ns  (1.000×)
Iter 1     262110 ns  (1.089×)  — memory_coalescing_swap_tile_dimensions

Final best: 263140 ns  (1.0849×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 2,   'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64},  num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 4096
-    xnumel = 31500
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    # X now covers spatial HW (xnumel=31500, stride-1 in output) -> coalesced stores
+    # Y now covers NC (ynumel=4096, stride-1 in input)
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # spatial HW index (warp-fast)
+    yoffset = tl.multiple_of(tl.program_id(1) * YBLOCK, YBLOCK)
+    yindex = tl.max_contiguous(yoffset + tl.arange(0, YBLOCK), YBLOCK)[:, None]  # NC index (slow)
+    xmask = (xindex < 31500) & (yindex < 4096)        # updated bounds
+    x2 = xindex          # spatial (stride-1 in output) -> COALESCED STORE
+    y0 = yindex % 128    # channel (stride-1 in input)
+    y1 = yindex // 128   # batch
+    y3 = yindex          # NC combined
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask)
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)  # COALESCED: stride=1 in X
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +47,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
