# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285090 ns — **Best latency**: 83100 ns (**3.4307×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_register_pressure_via_num_stages2_and_alias_inlining`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_056/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285090 ns
- **SM Throughput**: 20.23 % | **Memory**: 79.08 % | **DRAM**: 72.35 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 79.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalesced_transpose | 1.814× | 1.505× | ✓ |  |
| D | autotune_tile_sizes_for_memory_bound_transpose | 1.936× | 0.998× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285090 ns | 189470 ns | -95620 ns (-33.5%) | 285730 ns | +640 ns (+0.2%) |
| SM Throughput (%) | 20.23 % | 53.05 % | +32.82 pp | 20.26 % | +0.03 pp |
| Memory Throughput (%) | 79.08 % | 58.63 % | -20.45 pp | 78.90 % | -0.18 pp |
| DRAM Throughput (%) | 72.35 % | 26.19 % | -46.16 pp | 72.23 % | -0.12 pp |
| Achieved Occupancy (%) | 94.05 % | 93.90 % | -0.15 pp | 93.81 % | -0.24 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.83 % | 47.65 % | -32.18 pp | 79.58 % | -0.25 pp |
| IPC Active | 0.79 inst/cycle | 2.10 inst/cycle | +1.31 | 0.80 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalesced_transpose` (2.884×)

For a Triton transpose kernel where the load address has stride-1 along the Y-dimension (e.g., y0 = yindex % 128) and stride-128 along the X-dimension, adding autotune configs with YBLOCK=128 and small XBLOCK (8, 16, 32) aligns the thread tile with the 128-element contiguous segment in the load, enabling coalesced 128-bit LDG.128 instructions. This reduced L2 Cache Throughput from 45% to 21% and DRAM Throughput from 28% to 9%, while Warp Cycles Per Issued Instruction improved from 14.74 to 11.88 and Eligible Warps Per Scheduler rose from 0.82 to 0.97. The 2.88x speedup is measured relative to the original unoptimized baseline; the key causal change is the autotune config set expansion to YBLOCK=128 combined with switching the eviction policy from evict_last to evict_first (note: evict_first alone is harmful per item exp_evict_first_regression_iter4_005, but in this iteration the dominant effect is the YBLOCK=128 tile shape enabling coalesced loads).

> **Applicable when**: Triton transpose/layout-conversion kernel where the load address has stride-1 along the Y-dimension (e.g., y0 = yindex % N) and stride-N along the X-dimension; NCU reports CPIStall on L1TEX, IssueSlotUtilization, and SOLBottleneck; L2 Cache Throughput is elevated (>30%) with low L1 hit rate (<1%); existing autotune configs do not include YBLOCK >= the stride-1 segment size (e.g., YBLOCK < 128 when stride-1 segment is 128 elements).

> **Avoid when**: Do not apply when the load dimension is not stride-1 along Y (applying YBLOCK=128 to a strided Y-load does not help coalescing). Do not combine with evict_first eviction policy on the load when the kernel is not already coalesced — evict_first alone on strided loads causes severe regression (see item exp_evict_first_regression_iter4_005). Do not apply contiguity hints to the X-arange if X is strided in the load (only safe for the store-side X-axis).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_with_contiguity_hints | 1.816× | 1.504× | ✓ |  |
| B | shared_memory_tiled_transpose_for_dual_coalescing | 1.929× | 1.978× | ✓ |  |
| C | vectorized_stores_via_contiguity_hints_on_output_axis | 1.937× | 0.998× | ✓ |  |
| D | add_yblock128_configs_for_coalesced_transpose_loads | 2.679× | 2.848× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | shared_memory_til... | Δ | vectorized_stores... | Δ | add_yblock128_con... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285090 ns | 189570 ns | -95520 ns (-33.5%) | 144130 ns | -140960 ns (-49.4%) | 285660 ns | +570 ns (+0.2%) | 100100 ns | -184990 ns (-64.9%) |
| SM Throughput (%) | 20.23 % | 52.74 % | +32.51 pp | 48.31 % | +28.08 pp | 19.75 % | -0.48 pp | 51.98 % | +31.75 pp |
| Memory Throughput (%) | 79.08 % | 58.35 % | -20.73 pp | 42.27 % | -36.81 pp | 78.94 % | -0.14 pp | 59.69 % | -19.39 pp |
| DRAM Throughput (%) | 72.35 % | 26.27 % | -46.08 pp | 34.15 % | -38.20 pp | 72.22 % | -0.13 pp | 10.11 % | -62.24 pp |
| Achieved Occupancy (%) | 94.05 % | 93.94 % | -0.11 pp | 91.79 % | -2.26 pp | 93.99 % | -0.06 pp | 40.64 % | -53.41 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 27.00 register/thread | +2 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.83 % | 46.99 % | -32.84 pp | 53.68 % | -26.15 pp | 79.60 % | -0.23 pp | 46.74 % | -33.09 pp |
| IPC Active | 0.79 inst/cycle | 2.14 inst/cycle | +1.35 | 1.81 inst/cycle | +1.02 | 0.82 inst/cycle | +0.03 | 2.14 inst/cycle | +1.35 |


### Key Learnings

#### ✅ Plan A: `vectorized_stores_with_contiguity_hints` (1.504×)

When a Triton kernel's store index has a stride-1 pattern along the X-thread dimension (consecutive threads write to adjacent addresses), wrapping the tl.arange component with tl.max_contiguous(tl.multiple_of(..., XBLOCK), XBLOCK) hints enables the compiler to emit 128-bit vectorized store instructions. This targets the MemoryCacheAccessPattern rule (bytes-utilized-per-sector below 32) and reduces uncoalesced store sectors. Critically, these hints must only be applied to the store-side arange where addresses are genuinely contiguous — applying them to strided load indices causes kernel crashes (as seen in item 6ba05bf8).

> **Applicable when**: Triton kernel has a store index with stride-1 in the X-thread dimension (consecutive threads write adjacent addresses); NCU reports MemoryCacheAccessPattern with bytes-utilized-per-sector < 32 on stores; XBLOCK is a multiple of 4 (satisfying 128-bit alignment); UncoalescedGlobalAccess rule is triggered on stores.

> **Avoid when**: Do NOT apply contiguity hints to load indices that are strided (non-contiguous) — this causes kernel crashes. Do not apply when XBLOCK is not a multiple of 4 (breaks 128-bit alignment). Do not apply to the full xindex expression including xoffset; apply only to the tl.arange component.

---

## Iteration 3

**Best so far**: 2.8481× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_y_arange_contiguity_hints | 1.928× | 0.997× | ✓ |  |
| B | vectorized_loads_with_contiguity_hints_on_load_axis | 1.804× | 1.511× | ✓ |  |
| C | vectorized_loads_on_stride1_y_axis | 2.678× | 2.854× | ✓ |  |
| D | vectorized_loads_via_evict_first_cache_policy_for_streaming_transpose | 1.881× | 0.693× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285090 ns | 285890 ns | +800 ns (+0.3%) | 188700 ns | -96390 ns (-33.8%) | 99900 ns | -185190 ns (-65.0%) | 411140 ns | +126050 ns (+44.2%) |
| SM Throughput (%) | 20.23 % | 20.22 % | -0.01 pp | 52.31 % | +32.08 pp | 51.74 % | +31.51 pp | 14.00 % | -6.23 pp |
| Memory Throughput (%) | 79.08 % | 78.83 % | -0.25 pp | 57.91 % | -21.17 pp | 59.42 % | -19.66 pp | 60.04 % | -19.04 pp |
| DRAM Throughput (%) | 72.35 % | 72.17 % | -0.18 pp | 26.19 % | -46.16 pp | 10.09 % | -62.26 pp | 57.65 % | -14.70 pp |
| Achieved Occupancy (%) | 94.05 % | 94.00 % | -0.05 pp | 93.88 % | -0.17 pp | 42.75 % | -51.30 pp | 94.89 % | +0.84 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 32.00 register/thread | +7 | 27.00 register/thread | +2 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.83 % | 80.01 % | +0.18 pp | 47.36 % | -32.47 pp | 46.31 % | -33.52 pp | 86.33 % | +6.50 pp |
| IPC Active | 0.79 inst/cycle | 0.82 inst/cycle | +0.03 | 2.13 inst/cycle | +1.34 | 2.15 inst/cycle | +1.36 | 0.55 inst/cycle | -0.24 |

---

## Iteration 4

**Best so far**: 2.8481× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_alias_elimination_and_small_square_tiles | 3.265× | 3.445× | ✓ | **✓ best** |
| B | vectorized_stores_via_xarange_contiguity_hint_on_store_axis | 1.883× | 0.698× | ✓ |  |
| C | expand_autotune_configs_with_higher_stages_and_larger_yblock | 1.954× | 1.524× | ✓ |  |
| D | revert_evict_first_and_remove_index_aliases_to_restore_performance | 1.929× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | vectorized_stores... | Δ | expand_autotune_c... | Δ | revert_evict_firs... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285090 ns | 82750 ns | -202340 ns (-71.0%) | 408480 ns | +123390 ns (+43.3%) | 187040 ns | -98050 ns (-34.4%) | 289540 ns | +4450 ns (+1.6%) |
| SM Throughput (%) | 20.23 % | 49.64 % | +29.41 pp | 13.70 % | -6.53 pp | 51.86 % | +31.63 pp | 18.28 % | -1.95 pp |
| Memory Throughput (%) | 79.08 % | 33.03 % | -46.05 pp | 60.34 % | -18.74 pp | 57.40 % | -21.68 pp | 77.96 % | -1.12 pp |
| DRAM Throughput (%) | 72.35 % | 27.92 % | -44.43 pp | 57.92 % | -14.43 pp | 26.45 % | -45.90 pp | 71.28 % | -1.07 pp |
| Achieved Occupancy (%) | 94.05 % | 48.37 % | -45.68 pp | 94.87 % | +0.82 pp | 93.88 % | -0.17 pp | 93.91 % | -0.14 pp |
| Registers/Thread | 25.00 register/thread | 16.00 register/thread | -9 | 25.00 register/thread | +0 | 32.00 register/thread | +7 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.83 % | 47.35 % | -32.48 pp | 86.08 % | +6.25 pp | 46.85 % | -32.98 pp | 81.26 % | +1.43 pp |
| IPC Active | 0.79 inst/cycle | 2.05 inst/cycle | +1.26 | 0.55 inst/cycle | -0.24 | 2.14 inst/cycle | +1.35 | 0.75 inst/cycle | -0.04 |

---

## Iteration 5

**Best so far**: 3.4452× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_modulo_elimination_on_load_axis | 1.606× | 1.254× | ✓ |  |
| B | vectorized_loads_via_yblock128_load_coalescing | 2.669× | 2.884× | ✓ |  |
| C | reduce_register_pressure_via_num_stages2_and_alias_inlining | 3.279× | 3.431× | ✓ | **✓ best** |
| D | remove_evict_first_and_reduce_register_pressure_via_num_stages2_alias_inlining | 3.214× | 3.449× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | reduce_register_p... | Δ | remove_evict_firs... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285090 ns | 227360 ns | -57730 ns (-20.2%) | 98850 ns | -186240 ns (-65.3%) | 83100 ns | -201990 ns (-70.9%) | 82660 ns | -202430 ns (-71.0%) |
| SM Throughput (%) | 20.23 % | 63.34 % | +43.11 pp | 51.80 % | +31.57 pp | 48.59 % | +28.36 pp | 49.44 % | +29.21 pp |
| Memory Throughput (%) | 79.08 % | 67.38 % | -11.70 pp | 59.48 % | -19.60 pp | 32.94 % | -46.14 pp | 33.23 % | -45.85 pp |
| DRAM Throughput (%) | 72.35 % | 21.91 % | -50.44 pp | 9.48 % | -62.87 pp | 27.82 % | -44.53 pp | 27.98 % | -44.37 pp |
| Achieved Occupancy (%) | 94.05 % | 95.10 % | +1.05 pp | 43.03 % | -51.02 pp | 49.19 % | -44.86 pp | 49.20 % | -44.85 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 | 27.00 register/thread | +2 | 16.00 register/thread | -9 | 16.00 register/thread | -9 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.83 % | 36.05 % | -43.78 pp | 46.92 % | -32.91 pp | 48.19 % | -31.64 pp | 55.01 % | -24.82 pp |
| IPC Active | 0.79 inst/cycle | 2.52 inst/cycle | +1.73 | 2.14 inst/cycle | +1.35 | 2.08 inst/cycle | +1.29 | 1.89 inst/cycle | +1.10 |

---

## Cumulative Progress

```
Baseline:  285090 ns  (1.000×)
Iter 2     100100 ns  (2.848×)  — add_yblock128_configs_for_coalesced_transpose_loads
Iter 4     82750 ns  (3.445×)  — reduce_register_pressure_via_alias_elimination_and_small_square_tiles

Final best: 83100 ns  (3.4307×)
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
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    _yarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
+    yindex = yoffset + _yarange[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    _xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + _xarange[None, :]
     xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + ((yindex % 128) + 128*xindex + 4032000*(yindex // 128)), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (xindex + 31500*yindex), tmp0, xmask & ymask)
 
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
