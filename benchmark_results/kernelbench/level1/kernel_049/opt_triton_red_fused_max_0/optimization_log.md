# Kernel Optimization Log

## Summary

- **Kernel**: `max_dim1_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13610000 ns — **Best latency**: 2330000 ns (**5.8412×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`vectorized_loads_via_max_contiguous_multiple_of_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `max_dim1_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_049/synthesis/kernel_0_triton_red_fused_max_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (524160,1,1), Waves/SM: 496.36
- **Registers/thread**: 32
- **Duration**: 13610000 ns
- **SM Throughput**: 10.33 % | **Memory**: 95.32 % | **DRAM**: 12.89 %
- **Achieved Occupancy**: 97.65 %
- **IPC Active**: 0.41 inst/cycle | **No Eligible**: 89.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_xblock_tiling_and_pipelining | 1.667× | 0.882× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 13610000 ns | 15430000 ns | +1820000 ns (+13.4%) |
| SM Throughput (%) | 10.33 % | 9.68 % | -0.65 pp |
| Memory Throughput (%) | 95.32 % | 95.34 % | +0.02 pp |
| DRAM Throughput (%) | 12.89 % | 11.38 % | -1.51 pp |
| Achieved Occupancy (%) | 97.65 % | 69.29 % | -28.36 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 |
| Waves/SM | 496.36 | 661.82 | +165.46 |
| No Eligible (%) | 89.65 % | 90.32 % | +0.67 pp |
| IPC Active | 0.41 inst/cycle | 0.39 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints_and_transposed_tile_layout | 1.759× | 5.466× | ✓ |  |
| B | coalesced_access_via_warp_dimension_transposition | 0.872× | 0.882× | ✓ |  |
| C | coalesced_load_via_large_xblock_tiling | 1.687× | 5.510× | ✓ |  |
| D | coalesced_load_via_smem_transpose_reduction | 1.886× | 5.892× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.478× | 3.114× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | coalesced_access_... | Δ | coalesced_load_vi... | Δ | coalesced_load_vi... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13610000 ns | 2490000 ns | -11120000 ns (-81.7%) | 15430000 ns | +1820000 ns (+13.4%) | 2470000 ns | -11140000 ns (-81.9%) | 2310000 ns | -11300000 ns (-83.0%) | 4370000 ns | -9240000 ns (-67.9%) |
| SM Throughput (%) | 10.33 % | 33.71 % | +23.38 pp | 9.70 % | -0.63 pp | 36.27 % | +25.94 pp | 32.00 % | +21.67 pp | 16.92 % | +6.59 pp |
| Memory Throughput (%) | 95.32 % | 79.76 % | -15.56 pp | 95.33 % | +0.01 pp | 81.74 % | -13.58 pp | 80.83 % | -14.49 pp | 40.03 % | -55.29 pp |
| DRAM Throughput (%) | 12.89 % | 79.76 % | +66.87 pp | 11.37 % | -1.52 pp | 81.74 % | +68.85 pp | 80.83 % | +67.94 pp | 40.03 % | +27.14 pp |
| Achieved Occupancy (%) | 97.65 % | 69.10 % | -28.55 pp | 69.29 % | -28.36 pp | 37.10 % | -60.55 pp | 96.75 % | -0.90 pp | 36.88 % | -60.77 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 | 40.00 register/thread | +8 | 80.00 register/thread | +48 | 32.00 register/thread | +0 | 80.00 register/thread | +48 |
| Waves/SM | 496.36 | 2.59 | -493.77 | 661.82 | +165.46 | 20.68 | -475.68 | 7.76 | -488.60 | 5.17 | -491.19 |
| No Eligible (%) | 89.65 % | 65.79 % | -23.86 pp | 90.29 % | +0.64 pp | 62.79 % | -26.86 pp | 68.05 % | -21.60 pp | 81.17 % | -8.48 pp |
| IPC Active | 0.41 inst/cycle | 1.34 inst/cycle | +0.93 | 0.39 inst/cycle | -0.02 | 1.45 inst/cycle | +1.04 | 1.28 inst/cycle | +0.87 | 0.75 inst/cycle | +0.34 |

---

## Iteration 3

**Best so far**: 5.8918× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.886× | 5.841× | ✓ | **✓ best** |
| B | vectorized_loads_via_tl_max_contiguous_and_multiple_of_hints | 1.759× | 5.488× | ✓ |  |
| C | reduce_register_pressure_via_lower_num_stages_and_stronger_vectorization_hints | 1.760× | 6.215× | ✓ |  |
| D | vectorized_loads_via_transposed_tile_and_contiguous_hints | 1.760× | 5.444× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | reduce_register_p... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13610000 ns | 2330000 ns | -11280000 ns (-82.9%) | 2480000 ns | -11130000 ns (-81.8%) | 2190000 ns | -11420000 ns (-83.9%) | 2500000 ns | -11110000 ns (-81.6%) |
| SM Throughput (%) | 10.33 % | 31.72 % | +21.39 pp | 34.65 % | +24.32 pp | 39.78 % | +29.45 pp | 36.71 % | +26.38 pp |
| Memory Throughput (%) | 95.32 % | 80.31 % | -15.01 pp | 81.76 % | -13.56 pp | 86.63 % | -8.69 pp | 81.97 % | -13.35 pp |
| DRAM Throughput (%) | 12.89 % | 80.31 % | +67.42 pp | 81.76 % | +68.87 pp | 86.63 % | +73.74 pp | 81.97 % | +69.08 pp |
| Achieved Occupancy (%) | 97.65 % | 96.83 % | -0.82 pp | 36.70 % | -60.95 pp | 54.58 % | -43.07 pp | 55.42 % | -42.23 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 80.00 register/thread | +48 | 55.00 register/thread | +23 | 56.00 register/thread | +24 |
| Waves/SM | 496.36 | 7.76 | -488.60 | 10.34 | -486.02 | 6.89 | -489.47 | 13.79 | -482.57 |
| No Eligible (%) | 89.65 % | 68.12 % | -21.53 pp | 64.53 % | -25.12 pp | 60.08 % | -29.57 pp | 64.00 % | -25.65 pp |
| IPC Active | 0.41 inst/cycle | 1.27 inst/cycle | +0.86 | 1.39 inst/cycle | +0.98 | 1.62 inst/cycle | +1.21 | 1.45 inst/cycle | +1.04 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_max_contiguous_multiple_of_hints` (5.841×)

When a Triton reduction kernel is bottlenecked by shared memory bank conflicts (60-65% excessive wavefronts, 2.5-2.9-way conflicts) and L1TEX stalls, adding tl.max_contiguous and tl.multiple_of hints to the global offset vector and increasing num_stages from 2 to 3-4 produced no measurable improvement: all key metrics (DRAM throughput, L2 throughput, shared memory conflict rates, warp cycles per instruction, eligible warps per scheduler) remained essentially unchanged. The 5.84x speedup reported is attributable to the prior iteration's coalescing fix (XBLOCK tiling), not to these hints. The hints are compiler annotations that guide vectorized global loads but do not alter the shared memory layout used by tl.max's internal reduction, so bank conflicts in the smem reduction stage are unaffected.

> **Applicable when**: Never: this strategy does not work when the bottleneck is shared memory bank conflicts inside tl.max/tl.sum reductions. The hints only affect global load vectorization, not the compiler's internal smem layout for reductions.

> **Avoid when**: When NCU reports SharedMemoryConflicts with >50% excessive wavefronts as the primary bottleneck and global loads are already coalesced (DRAM throughput >75%). In this regime, tl.max_contiguous + tl.multiple_of hints and num_stages increases have no measurable effect on bank conflicts or kernel duration.

---

## Iteration 4

**Best so far**: 5.8918× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_smem_bank_conflicts_via_register_only_rowwise_accumulation | 1.327× | 4.321× | ✓ |  |
| B | eliminate_smem_bank_conflicts_via_explicit_register_reduction | 1.460× | 2.178× | ✓ |  |
| C | eliminate_2d_tile_smem_reduction_via_scalar_row_accumulation | 1.531× | 5.136× | ✓ |  |
| D | eliminate_smem_bank_conflicts_via_rowwise_register_accumulation | 1.537× | 3.256× | ✓ |  |
| E | add_triton_autotune | 1.358× | 5.296× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_smem_ba... | Δ | eliminate_smem_ba... | Δ | eliminate_2d_tile... | Δ | eliminate_smem_ba... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13610000 ns | 3150000 ns | -10460000 ns (-76.9%) | 6250000 ns | -7360000 ns (-54.1%) | 2650000 ns | -10960000 ns (-80.5%) | 4180000 ns | -9430000 ns (-69.3%) | 2570000 ns | -11040000 ns (-81.1%) |
| SM Throughput (%) | 10.33 % | 84.64 % | +74.31 pp | 97.31 % | +86.98 pp | 34.21 % | +23.88 pp | 85.53 % | +75.20 pp | 30.93 % | +20.60 pp |
| Memory Throughput (%) | 95.32 % | 56.70 % | -38.62 pp | 28.36 % | -66.96 pp | 80.10 % | -15.22 pp | 42.33 % | -52.99 pp | 80.94 % | -14.38 pp |
| DRAM Throughput (%) | 12.89 % | 56.70 % | +43.81 pp | 28.36 % | +15.47 pp | 80.10 % | +67.21 pp | 42.33 % | +29.44 pp | 80.94 % | +68.05 pp |
| Achieved Occupancy (%) | 97.65 % | 89.49 % | -8.16 pp | 94.30 % | -3.35 pp | 88.94 % | -8.71 pp | 94.65 % | -3.00 pp | 41.15 % | -56.50 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 72.00 register/thread | +40 |
| Waves/SM | 496.36 | 3.88 | -492.48 | 7.76 | -488.60 | 1.94 | -494.42 | 7.76 | -488.60 | 4.43 | -491.93 |
| No Eligible (%) | 89.65 % | 25.28 % | -64.37 pp | 34.20 % | -55.45 pp | 65.20 % | -24.45 pp | 13.82 % | -75.83 pp | 69.12 % | -20.53 pp |
| IPC Active | 0.41 inst/cycle | 2.99 inst/cycle | +2.58 | 2.63 inst/cycle | +2.22 | 1.41 inst/cycle | +1.00 | 3.45 inst/cycle | +3.04 | 1.23 inst/cycle | +0.82 |

---

## Iteration 5

**Best so far**: 5.8918× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_occupancy_via_larger_block_size_and_num_stages_1 | 1.381× | 5.296× | ✓ |  |
| B | pipeline_outer_loop_to_hide_l1tex_latency_and_align_grid_to_sm_count | 1.749× | 5.136× | ✓ |  |
| C | replace_static_range_scalar_loads_with_2d_tile_tl_max_reduction | 1.760× | 6.049× | ✓ |  |
| D | revert_scalar_row_loop_to_2d_tile_load_with_tl_max_axis0 | 1.873× | 6.049× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_occupanc... | Δ | pipeline_outer_lo... | Δ | replace_static_ra... | Δ | revert_scalar_row... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13610000 ns | 2570000 ns | -11040000 ns (-81.1%) | 2650000 ns | -10960000 ns (-80.5%) | 2250000 ns | -11360000 ns (-83.5%) | 2250000 ns | -11360000 ns (-83.5%) |
| SM Throughput (%) | 10.33 % | 30.47 % | +20.14 pp | 34.21 % | +23.88 pp | 40.29 % | +29.96 pp | 31.39 % | +21.06 pp |
| Memory Throughput (%) | 95.32 % | 81.04 % | -14.28 pp | 79.95 % | -15.37 pp | 85.94 % | -9.38 pp | 80.57 % | -14.75 pp |
| DRAM Throughput (%) | 12.89 % | 81.04 % | +68.15 pp | 79.95 % | +67.06 pp | 85.94 % | +73.05 pp | 80.57 % | +67.68 pp |
| Achieved Occupancy (%) | 97.65 % | 41.15 % | -56.50 pp | 88.96 % | -8.69 pp | 94.44 % | -3.21 pp | 96.95 % | -0.70 pp |
| Registers/Thread | 32.00 register/thread | 72.00 register/thread | +40 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 496.36 | 4.43 | -491.93 | 1.94 | -494.42 | 3.88 | -492.48 | 7.76 | -488.60 |
| No Eligible (%) | 89.65 % | 68.31 % | -21.34 pp | 64.48 % | -25.17 pp | 60.20 % | -29.45 pp | 68.53 % | -21.12 pp |
| IPC Active | 0.41 inst/cycle | 1.25 inst/cycle | +0.84 | 1.39 inst/cycle | +0.98 | 1.59 inst/cycle | +1.18 | 1.28 inst/cycle | +0.87 |

---

## Cumulative Progress

```
Baseline:  13610000 ns  (1.000×)
Iter 2     2310000 ns  (5.892×)  — coalesced_load_via_smem_transpose_reduction

Final best: 2330000 ns  (5.8412×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,13 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8, num_stages=1),
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
-        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1),
-        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
-        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
-        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=1),
-        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 2048}, num_warps=16, num_stages=1),
-        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'R0_BLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'R0_BLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'R0_BLOCK': 16},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 8},   num_warps=8, num_stages=2),
     ],
     key=['xnumel', 'r0_numel'],
 )
@@ -25,40 +22,47 @@

     XBLOCK: tl.constexpr,
     R0_BLOCK: tl.constexpr,
 ):
-    # x dimension: batch * dim2 = 128 * 4095 = 524160
-    # r0 dimension: dim1 = 4096 (reduction dimension)
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = xindex < xnumel
+    # NEW layout: each program covers a contiguous chunk of x0 (dim2, stride=1)
+    # for a fixed x1 (batch) index, and loops over r0 in R0_BLOCK chunks.
+    # This makes adjacent warp threads access adjacent x0 values (stride=1),
+    # enabling fully coalesced LDG.128 loads.
+    pid = tl.program_id(0)
+    num_x0_programs = tl.cdiv(4095, XBLOCK)
+    x1_index = pid // num_x0_programs   # batch index (scalar per program)
+    x0_offset = (pid % num_x0_programs) * XBLOCK
 
-    r0_base = tl.arange(0, R0_BLOCK)[None, :]
+    # 1-D range over x0 (dim2, stride=1 in memory)
+    x0_range = x0_offset + tl.arange(0, XBLOCK)  # [XBLOCK]
+    x0_mask = x0_range < 4095
 
-    # Decode x index: x0 = col index (dim2), x1 = batch index
-    x0 = xindex % 4095   # dim2 index
-    x1 = xindex // 4095  # batch index
+    # Apply contiguity hints to 1-D x0_range BEFORE broadcasting to 2-D
+    # This tells Triton the offsets are contiguous and 64-aligned -> LDG.128
+    # 64 divides all XBLOCK values (min XBLOCK=64), enables full LDG.128 vectorization
+    x0_hints = tl.max_contiguous(tl.multiple_of(x0_range, 64), 64)  # [XBLOCK]
 
-    # Initialize accumulator with -inf
-    _max_val = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
+    # Initialize accumulator with -inf, shape [XBLOCK]
+    _max_val = tl.full([XBLOCK], float('-inf'), tl.float32)
 
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
-        r0_index = r0_offset + r0_base
-        r0_mask = r0_index < r0_numel
+        r0_range = r0_offset + tl.arange(0, R0_BLOCK)  # [R0_BLOCK]
+        r0_mask = r0_range < r0_numel
 
-        # Input layout: [128, 4096, 4095] with strides [16773120, 4095, 1]
-        # ptr = x1 * 16773120 + r0_index * 4095 + x0
-        tmp0 = tl.load(
-            in_ptr0 + (x0 + 4095 * r0_index + 16773120 * x1),
-            r0_mask & xmask,
-            eviction_policy='evict_first',
-            other=float("-inf"),
-        )
-        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
-        _max_val = tl.where(r0_mask & xmask, tl.maximum(_max_val, tmp1), _max_val)
+        # 2-D tile: [R0_BLOCK, XBLOCK], x0 is fast (stride-1 in memory)
+        # ptr = x1_index * 16773120 + r0_range * 4095 + x0_range
+        ptrs = (in_ptr0
+                + x1_index * 16773120
+                + r0_range[:, None] * 4095
+                + x0_hints[None, :])  # broadcast 1-D hints to [R0_BLOCK, XBLOCK]
+        tile = tl.load(ptrs,
+                       mask=r0_mask[:, None] & x0_mask[None, :],
+                       other=float('-inf'),
+                       eviction_policy='evict_first')  # [R0_BLOCK, XBLOCK]
+        # Reduce over r0 dimension (axis=0), accumulate into [XBLOCK]
+        _max_val = tl.maximum(_max_val, tl.max(tile, axis=0))
 
-    # Reduce across R0_BLOCK dimension
-    result = tl.max(_max_val, axis=1)[:, None]
-    tl.store(out_ptr0 + xindex, result, xmask)
-
+    # Store result: output shape [128, 4095], stride [4095, 1]
+    out_ptrs = out_ptr0 + x1_index * 4095 + x0_range
+    tl.store(out_ptrs, _max_val, mask=x0_mask)
 
 def kernel_function(x):
     # x: [128, 4096, 4095], reduce along dim=1
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
