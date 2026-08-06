# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 4060 ns (**1.4828×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`contiguity_hints_for_vectorized_loads_in_layout_permutation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_056/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.80 % | **Memory**: 41.46 % | **DRAM**: 20.59 %
- **Achieved Occupancy**: 54.62 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 72.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.416× | 1.740× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_for_occupancy_and_coalescing | 1.400× | 1.760× | ✓ |  |
| D | add_autotune_increase_tile_sizes | 1.389× | 1.468× | ✓ |  |
| E | add_triton_autotune | 1.007× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3460 ns | -2560 ns (-42.5%) | 3420 ns | -2600 ns (-43.2%) | 4100 ns | -1920 ns (-31.9%) | 5950 ns | -70 ns (-1.2%) |
| SM Throughput (%) | 16.80 % | 16.69 % | -0.11 pp | 17.14 % | +0.34 pp | 19.95 % | +3.15 pp | 16.67 % | -0.13 pp |
| Memory Throughput (%) | 41.46 % | 19.75 % | -21.71 pp | 19.87 % | -21.59 pp | 22.94 % | -18.52 pp | 42.26 % | +0.80 pp |
| DRAM Throughput (%) | 20.59 % | 4.80 % | -15.79 pp | 4.80 % | -15.79 pp | 4.06 % | -16.53 pp | 20.70 % | +0.11 pp |
| Achieved Occupancy (%) | 54.62 % | 43.54 % | -11.08 pp | 45.14 % | -9.48 pp | 49.57 % | -5.05 pp | 55.17 % | +0.55 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.00 % | 54.55 % | -17.45 pp | 55.50 % | -16.50 pp | 51.47 % | -20.53 pp | 74.43 % | +2.43 pp |
| IPC Active | 1.15 inst/cycle | 1.68 inst/cycle | +0.53 | 1.63 inst/cycle | +0.48 | 1.81 inst/cycle | +0.66 | 1.15 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.7399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints_on_xindex_arange | 1.420× | 1.808× | ✓ |  |
| B | streaming_eviction_policy_for_one_shot_permutation | 1.355× | 1.808× | ✓ |  |
| C | contiguity_hints_for_vectorized_loads_in_layout_permutation | 1.423× | 1.483× | ✓ | **✓ best** |
| D | contiguity_hints_with_reduced_yblock_configs | 1.400× | 1.776× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | streaming_evictio... | Δ | contiguity_hints_... | Δ | contiguity_hints_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3330 ns | -2690 ns (-44.7%) | 3330 ns | -2690 ns (-44.7%) | 4060 ns | -1960 ns (-32.6%) | 3390 ns | -2630 ns (-43.7%) |
| SM Throughput (%) | 16.80 % | 17.65 % | +0.85 pp | 17.32 % | +0.52 pp | 18.92 % | +2.12 pp | 17.58 % | +0.78 pp |
| Memory Throughput (%) | 41.46 % | 20.68 % | -20.78 pp | 20.46 % | -21.00 pp | 21.79 % | -19.67 pp | 20.10 % | -21.36 pp |
| DRAM Throughput (%) | 20.59 % | 4.98 % | -15.61 pp | 4.98 % | -15.61 pp | 4.06 % | -16.53 pp | 4.88 % | -15.71 pp |
| Achieved Occupancy (%) | 54.62 % | 43.62 % | -11.00 pp | 44.42 % | -10.20 pp | 48.75 % | -5.87 pp | 43.78 % | -10.84 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.00 % | 55.14 % | -16.86 pp | 55.76 % | -16.24 pp | 51.69 % | -20.31 pp | 55.85 % | -16.15 pp |
| IPC Active | 1.15 inst/cycle | 1.68 inst/cycle | +0.53 | 1.66 inst/cycle | +0.51 | 1.85 inst/cycle | +0.70 | 1.68 inst/cycle | +0.53 |


### Key Learnings

#### ✅ Plan B: `streaming_eviction_policy_for_one_shot_permutation` (1.808×)

For streaming permutation kernels where each element is read and written exactly once (zero data reuse), switching tl.load eviction_policy from 'evict_last' to 'evict_first' and adding eviction_policy='evict_first' to tl.store reduces L2 pollution by immediately marking cache lines as lowest-priority after first use. The causal chain is: evict_first (.cs PTX operator) → L2 lines recycled sooner → reduced L2 slice imbalance (hottest slice above-average dropped from 20% to 15%) → slightly higher effective memory throughput (235→242 GB/s) → lower elapsed cycles (5125→4931). This change is zero-risk for correctness since eviction policy is a pure performance hint that does not affect loaded/stored values. Note: the profiling metrics show only modest improvement in this iteration because the kernel was already well-tuned by autotuning; the primary benefit is freeing L2 capacity for other concurrent kernels.

> **Applicable when**: Triton kernel performing a streaming memory operation (permutation, copy, type-cast) where each input element is loaded exactly once and each output element is stored exactly once (zero data reuse). L2 hit rate is anomalously high (>70%) for a streaming kernel, suggesting L2 pollution. WorkloadImbalance rule fires on L2 slices (hottest slice >15% above average). The load currently uses eviction_policy='evict_last' or no eviction hint.

> **Avoid when**: Kernel has data reuse (e.g., reduction, convolution with filter reuse, matrix multiply) where L2 caching of loaded data is beneficial. Do not apply if the kernel already uses evict_first. Do not apply if L2 hit rate is low (<40%), indicating data is not being cached anyway. Avoid if the kernel is compute-bound (SM throughput >70%) where memory policy has negligible effect.

#### ✅ Plan C: `contiguity_hints_for_vectorized_loads_in_layout_permutation` (1.483×)

Wrapping tl.arange(0, XBLOCK) with tl.max_contiguous(tl.multiple_of(..., XBLOCK), XBLOCK) before broadcasting is intended to signal alignment and contiguity to the Triton compiler, enabling 128-bit vectorized (LDG.128) global loads instead of scalar LDG.32. However, in this iteration the change actually *increased* kernel duration from 3.42 µs to 4.06 µs and raised register pressure from 18 to 25 registers/thread, while L1/TEX cache throughput rose from 37% to 54% — suggesting the compiler emitted more instructions rather than fewer. The uncoalesced access percentage and bytes-per-sector remained unchanged (50% excess sectors, 10.7/32 bytes/sector), indicating the store-side bottleneck was not addressed and the load-side vectorization did not materialize as expected.

> **Applicable when**: Never — this specific combination of contiguity hints on a 2D-tiled (XBLOCK x YBLOCK) permutation kernel increased register pressure and duration. The hint may be beneficial in purely 1D kernels where the arange is used directly without a [None, :] broadcast.

> **Avoid when**: When the arange is used in a 2D tile context (broadcast via [None, :] or [:, None]) — the contiguity hints interact poorly with the 2D indexing and increase register pressure. Also avoid when the kernel already has low register count (≤20 regs/thread) achieved by a prior autotune step, as the hints may undo that register reduction. Do not apply when UncoalescedGlobalAccess is on the store path (stride > 1 in output) — the hints only affect the load path and cannot fix store-side coalescing.

---

## Iteration 3

**Best so far**: 1.7399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_load_and_store_for_one_shot_permutation | 1.360× | 1.808× | ✓ |  |
| B | software_pipelining_via_num_stages_in_autotune_configs | 1.336× | 1.760× | ✓ |  |
| C | software_pipelining_via_num_stages_in_autotune_configs | 1.343× | 1.740× | ✓ |  |
| D | streaming_eviction_policy_for_one_shot_permutation | 1.335× | 1.427× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | software_pipelini... | Δ | software_pipelini... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3330 ns | -2690 ns (-44.7%) | 3420 ns | -2600 ns (-43.2%) | 3460 ns | -2560 ns (-42.5%) | 4220 ns | -1800 ns (-29.9%) |
| SM Throughput (%) | 16.80 % | 17.39 % | +0.59 pp | 17.28 % | +0.48 pp | 17.29 % | +0.49 pp | 19.96 % | +3.16 pp |
| Memory Throughput (%) | 41.46 % | 20.66 % | -20.80 pp | 19.99 % | -21.47 pp | 19.74 % | -21.72 pp | 22.21 % | -19.25 pp |
| DRAM Throughput (%) | 20.59 % | 4.99 % | -15.60 pp | 4.82 % | -15.77 pp | 4.76 % | -15.83 pp | 3.94 % | -16.65 pp |
| Achieved Occupancy (%) | 54.62 % | 49.16 % | -5.46 pp | 46.63 % | -7.99 pp | 44.50 % | -10.12 pp | 48.54 % | -6.08 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.00 % | 55.53 % | -16.47 pp | 55.81 % | -16.19 pp | 54.79 % | -17.21 pp | 50.21 % | -21.79 pp |
| IPC Active | 1.15 inst/cycle | 1.40 inst/cycle | +0.25 | 1.53 inst/cycle | +0.38 | 1.70 inst/cycle | +0.55 | 1.88 inst/cycle | +0.73 |


### Key Learnings

#### ✅ Plan D: `streaming_eviction_policy_for_one_shot_permutation` (1.808×)

For streaming permutation kernels where each element is read and written exactly once (zero data reuse), switching tl.load eviction_policy from 'evict_last' to 'evict_first' and adding eviction_policy='evict_first' to tl.store reduces L2 pollution by immediately marking cache lines as lowest-priority after first use. The causal chain is: evict_first (.cs PTX operator) → L2 lines recycled sooner → reduced L2 slice imbalance (hottest slice above-average dropped from 20% to 15%) → slightly higher effective memory throughput (235→242 GB/s) → lower elapsed cycles (5125→4931). This change is zero-risk for correctness since eviction policy is a pure performance hint that does not affect loaded/stored values. Note: the profiling metrics show only modest improvement in this iteration because the kernel was already well-tuned by autotuning; the primary benefit is freeing L2 capacity for other concurrent kernels.

> **Applicable when**: Triton kernel performing a streaming memory operation (permutation, copy, type-cast) where each input element is loaded exactly once and each output element is stored exactly once (zero data reuse). L2 hit rate is anomalously high (>70%) for a streaming kernel, suggesting L2 pollution. WorkloadImbalance rule fires on L2 slices (hottest slice >15% above average). The load currently uses eviction_policy='evict_last' or no eviction hint.

> **Avoid when**: Kernel has data reuse (e.g., reduction, convolution with filter reuse, matrix multiply) where L2 caching of loaded data is beneficial. Do not apply if the kernel already uses evict_first. Do not apply if L2 hit rate is low (<40%), indicating data is not being cached anyway. Avoid if the kernel is compute-bound (SM throughput >70%) where memory policy has negligible effect.

---

## Iteration 4

**Best so far**: 1.7399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staging_for_coalesced_stores | 1.259× | 1.776× | ✓ |  |
| B | fix_autotune_grid_lambda_and_expand_large_block_configs | 1.403× | 1.792× | ✓ |  |
| C | evict_first_store_plus_small_xblock_large_yblock_configs | 1.335× | 1.792× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | fix_autotune_grid... | Δ | evict_first_store... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3390 ns | -2630 ns (-43.7%) | 3360 ns | -2660 ns (-44.2%) | 3360 ns | -2660 ns (-44.2%) |
| SM Throughput (%) | 16.80 % | 15.85 % | -0.95 pp | 16.31 % | -0.49 pp | 17.09 % | +0.29 pp |
| Memory Throughput (%) | 41.46 % | 20.19 % | -21.27 pp | 20.41 % | -21.05 pp | 20.38 % | -21.08 pp |
| DRAM Throughput (%) | 20.59 % | 4.89 % | -15.70 pp | 4.95 % | -15.64 pp | 4.92 % | -15.67 pp |
| Achieved Occupancy (%) | 54.62 % | 44.82 % | -9.80 pp | 44.31 % | -10.31 pp | 48.91 % | -5.71 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.00 % | 55.58 % | -16.42 pp | 60.72 % | -11.28 pp | 55.47 % | -16.53 pp |
| IPC Active | 1.15 inst/cycle | 1.69 inst/cycle | +0.54 | 1.64 inst/cycle | +0.49 | 1.39 inst/cycle | +0.24 |


### Key Learnings

#### ⚠️ Plan A: `shared_memory_staging_for_coalesced_stores` (1.776×)

The plan intended to stage loads through shared memory to fix stride-3 store scatter (50% excess sectors, 10.7/32 bytes/sector), but the actual code change was only inserting tl.debug_barrier() between load and store plus setting num_stages=1 in all autotune configs — no actual shared memory allocation or transposition was implemented. As a result, the UncoalescedGlobalAccess excess sector percentage remained at 50% and bytes/sector stayed at 10.7, unchanged. The kernel duration actually increased slightly (3.33→3.39 µs) and achieved occupancy dropped (49.2%→44.8%), likely because num_stages=1 disabled software pipelining that was previously hiding memory latency. The reported 1.78x speedup reflects the cumulative baseline from prior iterations, not a gain from this change.

> **Applicable when**: Never — this pattern (debug_barrier only, no SMEM allocation) does not fix store-side uncoalesced access and regresses performance by disabling software pipelining.

> **Avoid when**: Any kernel where UncoalescedGlobalAccess fires on the store path due to strided output indexing. tl.debug_barrier() alone cannot change the memory access pattern; it only serializes warp execution. Setting num_stages=1 when the kernel previously used num_stages=2 or 3 removes software pipelining that hides memory latency, causing occupancy to drop and duration to increase.

#### ✅ Plan C: `evict_first_store_plus_small_xblock_large_yblock_configs` (1.792×)

Adding evict_first to the tl.store and expanding the autotune config space with small-XBLOCK/large-YBLOCK shapes (XBLOCK=32/64, YBLOCK=16/32) and additional num_stages variants produced no measurable improvement: elapsed cycles increased slightly from 4929 to 4992 and duration from 3.33 to 3.36 µs. All key metrics (eligible warps, warp cycles per instruction, L2 hit rate, memory throughput, uncoalesced access percentage, bytes per sector) remained essentially unchanged, indicating the autotuner selected a config equivalent to the prior best and the evict_first store hint had no measurable effect at this kernel state. The stride-3 store scatter bottleneck (50% excess sectors, 10.7/32 bytes/sector) is structural and cannot be resolved by tile shape or eviction policy changes alone.

> **Applicable when**: Triton layout-permutation kernel with stride-3 store scatter (50% excess sectors, ~10.7/32 bytes/sector), already autotuned with large-XBLOCK configs, where evict_first is absent on the store path and small-XBLOCK/large-YBLOCK configs have not been explored.

> **Avoid when**: The kernel is already at a well-tuned autotune baseline with evict_first on both load and store; the stride-3 store scatter bottleneck is structural and cannot be resolved by tile shape or eviction policy changes — further config expansion and evict_first store hints yield no measurable improvement. Do not invest budget here when UncoalescedGlobalAccess excess_sectors_pct remains at 50% and bytes_per_sector remains at 10.7 after prior eviction policy changes.

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3460 ns  (1.740×)  — vectorized_loads_with_contiguity_hints

Final best: 4060 ns  (1.4828×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 32}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 32,   'YBLOCK': 32}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=16, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +29,15 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

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
