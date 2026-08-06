# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 38050 ns — **Best latency**: 12610 ns (**3.0174×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fma_arithmetic_restructuring_for_non_fused_fp32_reduction`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_0_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 32
- **Duration**: 38050 ns
- **SM Throughput**: 44.90 % | **Memory**: 75.93 % | **DRAM**: 57.93 %
- **Achieved Occupancy**: 90.55 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 51.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.423× | 0.506× | ✓ |  |
| C | software_pipelining_with_autotune | 1.768× | 0.797× | ✓ |  |
| D | fix_uncoalesced_store_via_autotune_tile_sizes | 2.670× | 2.936× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipelini... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38050 ns | 75140 ns | +37090 ns (+97.5%) | 47740 ns | +9690 ns (+25.5%) | 12960 ns | -25090 ns (-65.9%) |
| SM Throughput (%) | 44.90 % | 73.22 % | +28.32 pp | 38.00 % | -6.90 pp | 68.74 % | +23.84 pp |
| Memory Throughput (%) | 75.93 % | 51.96 % | -23.97 pp | 56.81 % | -19.12 pp | 28.59 % | -47.34 pp |
| DRAM Throughput (%) | 57.93 % | 29.61 % | -28.32 pp | 23.91 % | -34.02 pp | 6.33 % | -51.60 pp |
| Achieved Occupancy (%) | 90.55 % | 34.46 % | -56.09 pp | 66.05 % | -24.50 pp | 65.36 % | -25.19 pp |
| Registers/Thread | 32.00 register/thread | 78.00 register/thread | +46 | 34.00 register/thread | +2 | 24.00 register/thread | -8 |
| Waves/SM | 7.42 | 19.80 | +12.38 | 9.90 | +2.48 | 7.42 | +0.00 |
| No Eligible (%) | 51.35 % | 24.00 % | -27.35 pp | 60.32 % | +8.97 pp | 15.65 % | -35.70 pp |
| IPC Active | 1.93 inst/cycle | 3.04 inst/cycle | +1.11 | 1.57 inst/cycle | -0.36 | 3.44 inst/cycle | +1.51 |

---

## Iteration 2

**Best so far**: 2.936× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_float_reciprocal_arithmetic | 2.688× | 2.865× | ✓ | **✓ best** |
| B | autotune_yblock32_for_coalesced_channels_last_stores | 1.131× | 2.929× | ✓ |  |
| D | add_triton_autotune | 2.686× | 2.922× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_fl... | Δ | autotune_yblock32... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38050 ns | 13280 ns | -24770 ns (-65.1%) | 12990 ns | -25060 ns (-65.9%) | 13020 ns | -25030 ns (-65.8%) |
| SM Throughput (%) | 44.90 % | 68.30 % | +23.40 pp | 68.52 % | +23.62 pp | 68.84 % | +23.94 pp |
| Memory Throughput (%) | 75.93 % | 28.37 % | -47.56 pp | 28.48 % | -47.45 pp | 28.58 % | -47.35 pp |
| DRAM Throughput (%) | 57.93 % | 6.18 % | -51.75 pp | 6.31 % | -51.62 pp | 6.31 % | -51.62 pp |
| Achieved Occupancy (%) | 90.55 % | 67.67 % | -22.88 pp | 64.62 % | -25.93 pp | 64.45 % | -26.10 pp |
| Registers/Thread | 32.00 register/thread | 24.00 register/thread | -8 | 24.00 register/thread | -8 | 24.00 register/thread | -8 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 51.35 % | 13.36 % | -37.99 pp | 14.28 % | -37.07 pp | 13.84 % | -37.51 pp |
| IPC Active | 1.93 inst/cycle | 3.44 inst/cycle | +1.51 | 3.38 inst/cycle | +1.45 | 3.41 inst/cycle | +1.48 |


### Key Learnings

#### ❌ Plan A: `fma_fusion_via_float_reciprocal_arithmetic` (2.865×)

Replacing an integer-literal reciprocal (tl.full([1,1], 1, tl.int32) / float) with a pure float reciprocal (tl.full([1,1], 1.0, tl.float32) / float) to enable FMA fusion produced no improvement and a slight regression (Duration: 12.96 µs → 13.28 µs). The executed instruction count was identical before and after (6,993,280), indicating the Triton compiler already optimized away the integer-to-float conversion and redundant multiply-by-1.0, and no new FMA chains were emitted. The NCU FPInstructions rule's estimated 9.3% global speedup was not realized because the compiler had already handled the arithmetic simplification, and the kernel's compute bottleneck (68.7% SM throughput) was not instruction-count limited.

> **Applicable when**: Never: this specific change (int→float literal for reciprocal) does not produce measurable gains even when NCU FPInstructions rule fires, because the Triton compiler already eliminates the int-to-float cast and redundant multiply-by-1.0 at the IR level.

> **Avoid when**: When the kernel is already compute-bound (Compute SM Throughput >65%) and the NCU FPInstructions rule fires but executed instruction count is identical before and after the change — this indicates the compiler has already optimized the arithmetic and the rule's speedup estimate is not achievable. Do not rely on NCU FPInstructions rule estimates when the Triton compiler is the code generator, as it performs its own FP simplification passes.

---

## Iteration 3

**Best so far**: 2.936× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_for_streaming_input_to_improve_l1_parameter_reuse | 2.602× | 2.936× | ✓ |  |
| B | streaming_eviction_policy_for_large_input_tensor | 2.592× | 2.959× | ✓ |  |
| C | increase_num_warps_to_close_occupancy_gap | 0.498× | 2.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_for_s... | Δ | streaming_evictio... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38050 ns | 12960 ns | -25090 ns (-65.9%) | 12860 ns | -25190 ns (-66.2%) | 12990 ns | -25060 ns (-65.9%) |
| SM Throughput (%) | 44.90 % | 68.88 % | +23.98 pp | 68.97 % | +24.07 pp | 69.74 % | +24.84 pp |
| Memory Throughput (%) | 75.93 % | 28.68 % | -47.25 pp | 28.70 % | -47.23 pp | 29.01 % | -46.92 pp |
| DRAM Throughput (%) | 57.93 % | 6.33 % | -51.60 pp | 6.38 % | -51.55 pp | 6.32 % | -51.61 pp |
| Achieved Occupancy (%) | 90.55 % | 63.59 % | -26.96 pp | 66.57 % | -23.98 pp | 70.74 % | -19.81 pp |
| Registers/Thread | 32.00 register/thread | 24.00 register/thread | -8 | 24.00 register/thread | -8 | 24.00 register/thread | -8 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 51.35 % | 13.55 % | -37.80 pp | 13.51 % | -37.84 pp | 14.00 % | -37.35 pp |
| IPC Active | 1.93 inst/cycle | 3.42 inst/cycle | +1.49 | 3.45 inst/cycle | +1.52 | 3.38 inst/cycle | +1.45 |


### Key Learnings

#### ❌ Plan A: `evict_first_for_streaming_input_to_improve_l1_parameter_reuse` (2.936×)

Changing the eviction policy of a large streaming input tensor load from evict_last to evict_first, with the intent of preserving L1 cache space for small frequently-reused parameter tensors (e.g., BN weight/bias/mean/var), produced no measurable improvement: L1/TEX Hit Rate remained at 19.33%, achieved occupancy actually dropped slightly (64.62% → 63.59%), and kernel duration barely changed (12.99 µs → 12.96 µs). The 2.936x reported speedup is the cumulative result from a prior iteration's tile-size optimization, not from this eviction-policy change. When a kernel is already compute-bound (Compute SM Throughput ~68%) and the occupancy gap is driven by warp scheduling overhead rather than L1 cache pollution, eviction policy hints on streaming loads do not move the needle on any profiling metric.

> **Applicable when**: Never: this specific change produced no improvement. The evict_first hint on a streaming load is theoretically sound but empirically ineffective when the kernel is already compute-bound (~68% SM throughput) and the L1 hit rate is constrained by factors other than eviction policy (e.g., tile size, access pattern, warp scheduling).

> **Avoid when**: When the kernel is compute-bound (Compute SM Throughput >65%) and L1/TEX Hit Rate does not change after applying evict_first — this indicates the L1 miss rate is not caused by streaming-data pollution of parameter cache lines. Also avoid when achieved occupancy is already below theoretical due to warp scheduling overhead rather than memory stalls, as eviction hints cannot address scheduling imbalances.

#### ✅ Plan B: `streaming_eviction_policy_for_large_input_tensor` (2.959×)

When a kernel loads a large input tensor (much larger than L2 cache) alongside small per-channel tensors that are reused across many spatial positions, using evict_last on the large tensor pollutes L2 with streaming data and creates L2 slice imbalance. Switching the large streaming tensor's load to eviction_policy='evict_first' (PTX .cs hint) signals the hardware to evict those lines immediately after use, freeing L2 capacity for the small reused tensors. In this case, the change reduced the no-eligible-warp rate from 14.28% to 13.51%, improved eligible warps per scheduler from 4.37 to 4.79, and raised achieved occupancy from 64.62% to 66.57%, yielding a modest but real latency reduction (12.99 µs → 12.86 µs). Note: the 2.96x reported speedup is cumulative from prior iterations; this specific change contributed ~1% duration improvement.

> **Applicable when**: Kernel loads a tensor much larger than L2 cache (e.g., >10x L2 size) in a streaming fashion (each element read once), alongside small tensors that are reused many times across the spatial loop. NCU AchievedOccupancy rule fires with a gap between theoretical and achieved occupancy not explained by register pressure. L2 slice imbalance is suspected (max L2 active cycles significantly above average). The large tensor is loaded with evict_last by default.

> **Avoid when**: The input tensor fits within L2 cache and benefits from retention (evict_last). All tensors are similarly sized and reused. The kernel is already at peak achieved occupancy. The tensor is written (stored), not loaded — eviction policy on stores has different semantics. Do not apply when the kernel's bottleneck is purely compute with no L2 pressure symptoms.

#### ✅ Plan C: `increase_num_warps_to_close_occupancy_gap` (2.929×)

Expanding the @triton.autotune config list to include higher num_warps (8, 16) and num_stages (2, 3) variants, plus smaller XBLOCK=4 configs, increased achieved occupancy from 64.6% to 70.7% and active warps per scheduler from 10.34 to 10.74, with a marginal improvement in SM throughput (68.5% → 69.7%). However, the kernel duration remained essentially unchanged (12.99 µs both before and after), indicating the occupancy gap was not the primary performance limiter at this stage — the kernel was already compute-bound and the additional warp headroom did not translate to wall-clock improvement. The 2.93x reported speedup is cumulative from prior iterations (tile coalescing), not from this specific change.

> **Applicable when**: AchievedOccupancy rule fires with achieved occupancy <70% vs theoretical 100%; scheduler No-Eligible cycles >10%; kernel is NOT yet at its compute roofline ceiling; the occupancy gap is caused by warp starvation rather than register or shared memory limits; block_limit_registers > block_limit_warps (register pressure is not the binding constraint).

> **Avoid when**: Kernel is already compute-bound at its roofline ceiling (Compute SM Throughput >68% with no memory bottleneck) and duration does not change despite occupancy improvements — this indicates the bottleneck is instruction throughput, not warp scheduling. Also avoid when register pressure is already the binding occupancy constraint (block_limit_registers < block_limit_warps), as adding higher num_warps configs will not help and may increase register spill risk.

---

## Iteration 4

**Best so far**: 2.936× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_div_with_rsqrt_for_fma_fusion | 2.751× | 3.034× | ✓ | **✓ best** |
| B | rsqrt_fusion_to_reduce_nonfused_fp32_instructions | 2.500× | 3.049× | ✓ |  |
| D | add_triton_autotune | 2.719× | 2.893× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_div_... | Δ | rsqrt_fusion_to_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38050 ns | 12540 ns | -25510 ns (-67.0%) | 12480 ns | -25570 ns (-67.2%) | 13150 ns | -24900 ns (-65.4%) |
| SM Throughput (%) | 44.90 % | 56.68 % | +11.78 pp | 57.31 % | +12.41 pp | 68.48 % | +23.58 pp |
| Memory Throughput (%) | 75.93 % | 29.70 % | -46.23 pp | 30.07 % | -45.86 pp | 28.44 % | -47.49 pp |
| DRAM Throughput (%) | 57.93 % | 6.55 % | -51.38 pp | 6.57 % | -51.36 pp | 6.23 % | -51.70 pp |
| Achieved Occupancy (%) | 90.55 % | 40.92 % | -49.63 pp | 39.86 % | -50.69 pp | 68.52 % | -22.03 pp |
| Registers/Thread | 32.00 register/thread | 24.00 register/thread | -8 | 22.00 register/thread | -10 | 24.00 register/thread | -8 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 51.35 % | 26.33 % | -25.02 pp | 26.32 % | -25.03 pp | 13.33 % | -38.02 pp |
| IPC Active | 1.93 inst/cycle | 2.91 inst/cycle | +0.98 | 2.93 inst/cycle | +1.00 | 3.43 inst/cycle | +1.50 |

---

## Iteration 5

**Best so far**: 3.0343× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_for_batchnorm_arithmetic_chain | 2.648× | 3.042× | ✓ |  |
| B | fma_arithmetic_restructuring_for_non_fused_fp32_reduction | 2.755× | 3.017× | ✓ | **✓ best** |
| C | replace_two_step_sqrt_div_with_libdevice_rsqrt | 2.718× | 3.034× | ✓ |  |
| D | add_triton_autotune | 2.661× | 3.049× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_for_... | Δ | fma_arithmetic_re... | Δ | replace_two_step_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38050 ns | 12510 ns | -25540 ns (-67.1%) | 12610 ns | -25440 ns (-66.9%) | 12540 ns | -25510 ns (-67.0%) | 12480 ns | -25570 ns (-67.2%) |
| SM Throughput (%) | 44.90 % | 56.34 % | +11.44 pp | 55.86 % | +10.96 pp | 56.33 % | +11.43 pp | 56.77 % | +11.87 pp |
| Memory Throughput (%) | 75.93 % | 29.49 % | -46.44 pp | 29.27 % | -46.66 pp | 29.51 % | -46.42 pp | 29.72 % | -46.21 pp |
| DRAM Throughput (%) | 57.93 % | 6.56 % | -51.37 pp | 6.49 % | -51.44 pp | 6.54 % | -51.39 pp | 6.56 % | -51.37 pp |
| Achieved Occupancy (%) | 90.55 % | 41.08 % | -49.47 pp | 41.40 % | -49.15 pp | 40.85 % | -49.70 pp | 39.81 % | -50.74 pp |
| Registers/Thread | 32.00 register/thread | 22.00 register/thread | -10 | 24.00 register/thread | -8 | 24.00 register/thread | -8 | 22.00 register/thread | -10 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 51.35 % | 25.81 % | -25.54 pp | 26.91 % | -24.44 pp | 26.79 % | -24.56 pp | 26.76 % | -24.59 pp |
| IPC Active | 1.93 inst/cycle | 2.99 inst/cycle | +1.06 | 2.93 inst/cycle | +1.00 | 2.90 inst/cycle | +0.97 | 2.94 inst/cycle | +1.01 |


### Key Learnings

#### ❌ Plan A: `explicit_fma_for_batchnorm_arithmetic_chain` (3.042×)

Replacing a two-instruction sequence (FMUL + FADD) in the batch normalization normalization chain with an explicit tl.math.fma() call produced no measurable improvement: executed instruction count was identical before and after (5,597,760), duration barely changed (12.48 µs → 12.51 µs, a slight regression), and all key metrics (warp cycles per issued instruction, eligible warps per scheduler, No-Eligible rate) were statistically unchanged. The NCU FPInstructions rule's estimated 7.3% global speedup was not realized because the Triton compiler (via LLVM/PTX lowering) already emits FFMA for eligible FMUL+FADD pairs, making the explicit tl.math.fma() call redundant. When executed instruction count does not decrease after applying tl.math.fma(), the compiler had already performed the fusion and no further gain is possible from this direction.

> **Applicable when**: Never for this specific pattern: when the NCU FPInstructions rule fires but executed instruction count does not decrease after applying tl.math.fma(), the Triton compiler has already fused the FMUL+FADD pair into FFMA at the PTX level. Do not apply tl.math.fma() as a follow-up optimization after libdevice.rsqrt has already reduced instruction count, as the remaining non-fused instructions may be in boundary/predicated tiles that the compiler cannot fuse regardless.

> **Avoid when**: When executed instruction count is identical before and after the change — this is the definitive signal that the compiler already performed the fusion. Also do not apply when the kernel is in a low-occupancy regime (Achieved Occupancy ~40%) driven by warp scheduling overhead rather than instruction throughput, as FMA fusion cannot address scheduling stalls. Do not apply when the FPInstructions rule fires alongside a large No-Eligible warp rate (>25%) — the bottleneck is warp availability, not instruction count.

---

## Cumulative Progress

```
Baseline:  38050 ns  (1.000×)
Iter 1     12960 ns  (2.936×)  — fix_uncoalesced_store_via_autotune_tile_sizes
Iter 4     12540 ns  (3.034×)  — replace_sqrt_div_with_rsqrt_for_fma_fusion

Final best: 12610 ns  (3.0174×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 32}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 32}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 32}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024,'YBLOCK': 32}, num_warps=8,  num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 320
@@ -19,27 +44,20 @@

     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1, 1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1, 1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp8 = libdevice.rsqrt(tmp5)
+    tmp_scale = tmp8 * tmp12
+    tmp15 = tl.math.fma(tmp_scale, tmp2, tmp14)
     tmp16 = tl.full([1, 1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(out_ptr0 + (y0 + 32*x2 + 1605632*y1), tmp17, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 32*x2 + 1605632*y1), tmp17, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -51,7 +69,7 @@

     ynumel = 320
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 320, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 320, 50176)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
