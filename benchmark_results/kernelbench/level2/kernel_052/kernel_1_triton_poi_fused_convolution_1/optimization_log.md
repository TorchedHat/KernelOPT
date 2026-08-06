# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286460 ns — **Best latency**: 286020 ns (**1.0015×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restore_baseline_and_fuse_bias_add_into_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_052/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286460 ns
- **SM Throughput**: 19.78 % | **Memory**: 78.63 % | **DRAM**: 72.02 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 0.999× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286460 ns | 285630 ns | -830 ns (-0.3%) |
| SM Throughput (%) | 19.78 % | 19.64 % | -0.14 pp |
| Memory Throughput (%) | 78.63 % | 78.86 % | +0.23 pp |
| DRAM Throughput (%) | 72.02 % | 72.24 % | +0.22 pp |
| Achieved Occupancy (%) | 94.04 % | 94.04 % | +0.00 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 80.31 % | +0.00 pp |
| IPC Active | 0.79 inst/cycle | 0.78 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_tile_dimensions | 1.000× | 0.999× | ✓ | **✓ best** |
| D | eviction_policy_fix_and_larger_yblock | 0.773× | 0.719× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | eviction_policy_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 286750 ns | +290 ns (+0.1%) | 398370 ns | +111910 ns (+39.1%) |
| SM Throughput (%) | 19.78 % | 19.17 % | -0.61 pp | 14.08 % | -5.70 pp |
| Memory Throughput (%) | 78.63 % | 78.61 % | -0.02 pp | 61.72 % | -16.91 pp |
| DRAM Throughput (%) | 72.02 % | 71.95 % | -0.07 pp | 59.02 % | -13.00 pp |
| Achieved Occupancy (%) | 94.04 % | 93.39 % | -0.65 pp | 94.88 % | +0.84 pp |
| Registers/Thread | 25.00 register/thread | 24.00 register/thread | -1 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 80.77 % | +0.46 pp | 85.77 % | +5.46 pp |
| IPC Active | 0.79 inst/cycle | 0.76 inst/cycle | -0.03 | 0.56 inst/cycle | -0.23 |


### Key Learnings

#### ❌ Plan D: `eviction_policy_fix_and_larger_yblock` (0.720×)

Switching tl.load eviction_policy from 'evict_last' to 'evict_first' on a strided transpose kernel caused a 28% slowdown (286 µs → 398 µs). Despite the intent to reduce L2 pollution, the L2 hit rate dropped from 66.6% to 60.7%, DRAM throughput fell from 72% to 59%, and warp cycles per issued instruction worsened from 78 to 110 cycles — indicating that evict_first reduced effective cache reuse and increased memory latency. The scheduler 'No Eligible' fraction increased from 80.8% to 86.2%, meaning warps stalled even more severely after the change. This confirms that even in a strided transpose with 0% L1 hit rate, the L2 does provide meaningful (~67%) hit-rate reuse that evict_first destroys.

> **Applicable when**: Never — switching evict_last to evict_first on this strided transpose access pattern is harmful.

> **Avoid when**: Strided transpose kernels where L2 hit rate is already ~60-67% and DRAM is the bottleneck. Even though L1 hit rate is 0%, the L2 provides meaningful reuse for the y-dimension grouping (128 elements sharing the same y1 group). Switching to evict_first destroys this L2 reuse, lowers DRAM throughput utilization, and increases warp stall cycles. Do not apply evict_first when L2 hit rate > 50% and the kernel is already memory-bound with high occupancy (93%+) and many waves per SM (59+).

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_contiguous_y_dimension | 0.396× | 0.390× | ✓ |  |
| B | software_pipelined_evict_first_transpose | 0.714× | 0.720× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 734590 ns | +448130 ns (+156.4%) | 397860 ns | +111400 ns (+38.9%) |
| SM Throughput (%) | 19.78 % | 8.01 % | -11.77 pp | 13.41 % | -6.37 pp |
| Memory Throughput (%) | 78.63 % | 94.09 % | +15.46 pp | 61.83 % | -16.80 pp |
| DRAM Throughput (%) | 72.02 % | 28.17 % | -43.85 pp | 58.93 % | -13.09 pp |
| Achieved Occupancy (%) | 94.04 % | 79.30 % | -14.74 pp | 94.36 % | +0.32 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 | 24.00 register/thread | -1 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 91.94 % | +11.63 pp | 86.18 % | +5.87 pp |
| IPC Active | 0.79 inst/cycle | 0.32 inst/cycle | -0.47 | 0.55 inst/cycle | -0.24 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_contiguous_y_dimension` (0.390×)

Reordering tile dimensions to YBLOCK=128, XBLOCK=1 with tl.max_contiguous/tl.multiple_of hints on the Y-axis caused a 2.56x slowdown (287 µs → 735 µs). While L1 hit rate improved dramatically from 0% to 71.6% and L2 throughput dropped from 96.8% to 51%, the achieved occupancy fell from 93.4% to 79.3%, registers per thread increased from 24 to 28, and warp cycles per issued instruction doubled from 78 to 158 — indicating that the XBLOCK=1 tiling created severe MIO pipeline pressure (CPIStall shifted from L1TEX stalls to MIO queue full stalls at 63.1 cycles) and the uncoalesced global access rate exploded from 2% to 78% of sectors. The dimension reorder fixed the load coalescing but broke the store coalescing catastrophically, and the reduced occupancy could not hide the resulting latency.

> **Applicable when**: Never — this specific combination is harmful for this transpose pattern.

> **Avoid when**: Transpose kernels where the output store index has the form (x2 + large_stride*y3): making XBLOCK=1 means x2 is a scalar per CTA, so all 128 threads in the Y-warp write to the same x2 column with stride large_stride between them — this makes stores completely uncoalesced (78% excessive sectors vs 2% before). Do not reorder to YBLOCK=128/XBLOCK=1 when the store pattern is (scalar_x + large_stride*vector_y), as this trades load coalescing for store uncoalescing. The MIO queue full stall (63.1 cycles, 39.9% of total stall) also indicates that XBLOCK=1 with YBLOCK=128 creates excessive shared memory or special instruction pressure. Additionally, reduced occupancy (79.3% vs 93.4%) from increased register pressure (24→28 regs/thread) cannot hide the worsened memory latency.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_x_dimension_store_hints | 0.999× | 1.004× | ✓ |  |
| C | vectorized_stores_contiguous_x_dimension | 0.396× | 0.390× | ✓ |  |
| D | restore_baseline_and_fuse_bias_add_into_transpose | 1.000× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_stores... | Δ | restore_baseline_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 285410 ns | -1050 ns (-0.4%) | 734400 ns | +447940 ns (+156.4%) | 286020 ns | -440 ns (-0.2%) |
| SM Throughput (%) | 19.78 % | 19.14 % | -0.64 pp | 8.01 % | -11.77 pp | 19.18 % | -0.60 pp |
| Memory Throughput (%) | 78.63 % | 78.91 % | +0.28 pp | 94.13 % | +15.50 pp | 78.75 % | +0.12 pp |
| DRAM Throughput (%) | 72.02 % | 72.28 % | +0.26 pp | 28.12 % | -43.90 pp | 72.14 % | +0.12 pp |
| Achieved Occupancy (%) | 94.04 % | 93.39 % | -0.65 pp | 80.39 % | -13.65 pp | 93.39 % | -0.65 pp |
| Registers/Thread | 25.00 register/thread | 24.00 register/thread | -1 | 28.00 register/thread | +3 | 24.00 register/thread | -1 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 80.78 % | +0.47 pp | 91.93 % | +11.62 pp | 80.52 % | +0.21 pp |
| IPC Active | 0.79 inst/cycle | 0.76 inst/cycle | -0.03 | 0.32 inst/cycle | -0.47 | 0.77 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan C: `vectorized_stores_contiguous_x_dimension` (0.390×)

Adding tl.max_contiguous and tl.multiple_of hints to both the X and Y arange dimensions (along with a cache_modifier='.cs' store hint) caused a 2.56x slowdown (398 µs → 734 µs). While L1 hit rate improved from 0.01% to 71.88% and L2 throughput dropped from 88.4% to 51%, the achieved occupancy fell from 94.4% to 80.4%, registers per thread increased from 24 to 28, and warp cycles per issued instruction rose from 110 to 161 cycles. The CPIStall reason shifted from L1TEX stalls to MIO queue full stalls (65 cycles, 40.5% of total), and uncoalesced global access exploded from 2% to 78% of sectors — indicating the hints caused the compiler to generate a fundamentally different (and worse) access pattern rather than simply vectorizing the existing stride-1 stores.

> **Applicable when**: Never — this specific combination of hints is harmful for this transpose pattern.

> **Avoid when**: Transpose kernels where YBLOCK is small (e.g. 8) and tl.multiple_of hint value (128) does not match YBLOCK: applying tl.multiple_of(yoffset + arange(0, YBLOCK), 128) when YBLOCK=8 is mathematically invalid (yoffset is a multiple of 8, not 128), causing the compiler to generate incorrect or severely degraded access patterns. The uncoalesced global access rate exploded from 2% to 78% of sectors, and MIO queue full stalls appeared (65 cycles, 40.5% of stall budget) where none existed before. Additionally, adding cache_modifier='.cs' (streaming) to stores in a kernel that is already L2-bound (88% L2 throughput) can reduce effective L2 reuse. Do not apply tl.max_contiguous/tl.multiple_of hints when the hint alignment value does not match the actual tile dimension size, and do not combine X-axis and Y-axis hints simultaneously when only one axis is genuinely contiguous and aligned.

---

## Iteration 5

**Best so far**: 1.0015× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_wider_yblock_for_128bit_channel_loads | 0.396× | 0.389× | ✓ |  |
| B | vectorized_loads_x_only_correct_alignment | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 735580 ns | +449120 ns (+156.8%) | 286880 ns | +420 ns (+0.1%) |
| SM Throughput (%) | 19.78 % | 8.01 % | -11.77 pp | 19.50 % | -0.28 pp |
| Memory Throughput (%) | 78.63 % | 94.10 % | +15.47 pp | 78.47 % | -0.16 pp |
| DRAM Throughput (%) | 72.02 % | 28.08 % | -43.94 pp | 71.93 % | -0.09 pp |
| Achieved Occupancy (%) | 94.04 % | 80.99 % | -13.05 pp | 93.39 % | -0.65 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 | 24.00 register/thread | -1 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 91.94 % | +11.63 pp | 80.49 % | +0.18 pp |
| IPC Active | 0.79 inst/cycle | 0.32 inst/cycle | -0.47 | 0.77 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  286460 ns  (1.000×)
Iter 4     286020 ns  (1.002×)  — restore_baseline_and_fuse_bias_add_into_transpose

Final best: 286020 ns  (1.0015×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
+    xoffset = tl.program_id(0) * XBLOCK
+    # Apply hints on 1D arange BEFORE reshape — xoffset is always a multiple of XBLOCK=128
+    _xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = _xrange[:, None]   # [XBLOCK, 1]
+    xmask = xindex < xnumel
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
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
+    # Apply hints on 1D arange BEFORE reshape — yoffset is always a multiple of YBLOCK=8
+    _yrange = tl.max_contiguous(tl.multiple_of(yoffset + tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
+    yindex = _yrange[None, :]   # [1, YBLOCK]
+    ymask = tl.full([YBLOCK], True, tl.int1)[None, :]
+    x2 = xindex                 # [XBLOCK, 1]
+    y0 = (yindex % 128)         # [1, YBLOCK] stride-1 in input
+    y1 = yindex // 128          # [1, YBLOCK]
+    y3 = yindex                 # [1, YBLOCK]
+    combined_mask = xmask & ymask
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=combined_mask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=combined_mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
