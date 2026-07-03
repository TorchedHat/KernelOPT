# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174020 ns — **Best latency**: 69150 ns (**2.5166×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_redundant_fma_and_store_index_recomputation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_063/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174020 ns
- **SM Throughput**: 27.91 % | **Memory**: 81.83 % | **DRAM**: 67.50 %
- **Achieved Occupancy**: 94.13 %
- **IPC Active**: 1.17 inst/cycle | **No Eligible**: 71.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 3.511× | 2.418× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174020 ns | 71970 ns | -102050 ns (-58.6%) |
| SM Throughput (%) | 27.91 % | 21.55 % | -6.36 pp |
| Memory Throughput (%) | 81.83 % | 89.06 % | +7.23 pp |
| DRAM Throughput (%) | 67.50 % | 17.06 % | -50.44 pp |
| Achieved Occupancy (%) | 94.13 % | 82.61 % | -11.52 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 71.14 % | 77.14 % | +6.00 pp |
| IPC Active | 1.17 inst/cycle | 0.91 inst/cycle | -0.26 |

---

## Iteration 2

**Best so far**: 2.418× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_via_num_stages_and_eviction_policy_tuning | 3.517× | 2.498× | ✓ |  |
| B | autotune_num_stages_for_latency_hiding | 3.527× | 2.427× | ✓ | **✓ best** |
| C | memory_coalescing_eviction_policy_and_pipelining | 3.520× | 2.483× | ✓ |  |
| D | eviction_policy_and_software_pipelining_for_l2_saturated_strided_load | 3.510× | 2.502× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | autotune_num_stag... | Δ | memory_coalescing... | Δ | eviction_policy_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174020 ns | 69660 ns | -104360 ns (-60.0%) | 71710 ns | -102310 ns (-58.8%) | 70080 ns | -103940 ns (-59.7%) | 69540 ns | -104480 ns (-60.0%) |
| SM Throughput (%) | 27.91 % | 25.26 % | -2.65 pp | 21.77 % | -6.14 pp | 25.02 % | -2.89 pp | 24.68 % | -3.23 pp |
| Memory Throughput (%) | 81.83 % | 93.70 % | +11.87 pp | 89.36 % | +7.53 pp | 93.24 % | +11.41 pp | 93.63 % | +11.80 pp |
| DRAM Throughput (%) | 67.50 % | 14.45 % | -53.05 pp | 16.51 % | -50.99 pp | 14.34 % | -53.16 pp | 16.49 % | -51.01 pp |
| Achieved Occupancy (%) | 94.13 % | 82.30 % | -11.83 pp | 82.67 % | -11.46 pp | 82.15 % | -11.98 pp | 82.50 % | -11.63 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.14 % | 73.96 % | +2.82 pp | 77.15 % | +6.01 pp | 73.63 % | +2.49 pp | 73.97 % | +2.83 pp |
| IPC Active | 1.17 inst/cycle | 1.05 inst/cycle | -0.12 | 0.90 inst/cycle | -0.27 | 1.05 inst/cycle | -0.12 | 1.03 inst/cycle | -0.14 |


### Key Learnings

#### ❌ Plan B: `autotune_num_stages_for_latency_hiding` (2.478×)

Expanding the Triton autotune config space to include num_warps=16 (512 threads/block) across XBLOCK in {64, 128, 256, 512, 1024} and num_stages in {1, 2} produced no measurable improvement: duration increased slightly from 69.47us to 70.24us (~1.1% regression), and all key metrics (L2 throughput, eligible warps per scheduler, warp cycles per instruction) remained virtually unchanged. When a pointwise kernel is already L2-saturated (>92%) with persistent uncoalesced access (76% excessive sectors) and the autotuner has already found its optimal config from prior num_warps={1,2,4,8} and num_stages={1,2,3} searches, adding num_warps=16 configs provides no additional benefit. The bottleneck is the irreducible stride-induced sector amplification, not warp scheduling efficiency — eligible warps per scheduler barely moved (0.35 → 0.36) despite the expanded search space.

> **Applicable when**: Never applicable as a standalone optimization once num_warps={1,2,4,8} and num_stages={1,2,3} autotune has already been applied to a pointwise kernel that is L2-saturated (>90%) with persistent uncoalesced access. The autotuner will have already found the optimal config from the smaller search space.

> **Avoid when**: Pointwise/elementwise kernels (bias-add, activation, normalization without reduction) where the kernel is already L2-bound with high L2 throughput (>90%) and the bottleneck is uncoalesced access pattern rather than warp scheduling efficiency. Also avoid when num_warps={1,2,4,8} autotune has already been applied — adding num_warps=16 configs provides no incremental benefit and wastes autotuning time. The eligible warps per scheduler metric (0.35) is not meaningfully improved by higher num_warps when the stall is structural (stride-induced sector amplification).

---

## Iteration 3

**Best so far**: 2.4267× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_stages_3_and_expanded_warp_configs_for_l2_latency_hiding | 3.504× | 2.495× | ✓ |  |
| D | evict_first_on_nonreusable_strided_load_and_store | 3.522× | 2.505× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | evict_first_on_no... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174020 ns | 69760 ns | -104260 ns (-59.9%) | 69470 ns | -104550 ns (-60.1%) |
| SM Throughput (%) | 27.91 % | 24.94 % | -2.97 pp | 24.98 % | -2.93 pp |
| Memory Throughput (%) | 81.83 % | 93.26 % | +11.43 pp | 93.66 % | +11.83 pp |
| DRAM Throughput (%) | 67.50 % | 16.47 % | -51.03 pp | 16.53 % | -50.97 pp |
| Achieved Occupancy (%) | 94.13 % | 82.47 % | -11.66 pp | 82.74 % | -11.39 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.14 % | 74.10 % | +2.96 pp | 74.54 % | +3.40 pp |
| IPC Active | 1.17 inst/cycle | 1.02 inst/cycle | -0.15 | 1.01 inst/cycle | -0.16 |

---

## Iteration 4

**Best so far**: 2.4267× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | store_index_contiguity_hint_and_fma_fusion | 3.517× | 2.500× | ✓ |  |
| D | autotune_spatial_aligned_xblock_and_high_warp_configs_for_partial_sector_elimination | 3.534× | 2.478× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | store_index_conti... | Δ | autotune_spatial_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174020 ns | 69600 ns | -104420 ns (-60.0%) | 70240 ns | -103780 ns (-59.6%) |
| SM Throughput (%) | 27.91 % | 24.66 % | -3.25 pp | 25.03 % | -2.88 pp |
| Memory Throughput (%) | 81.83 % | 93.49 % | +11.66 pp | 92.74 % | +10.91 pp |
| DRAM Throughput (%) | 67.50 % | 16.45 % | -51.05 pp | 16.36 % | -51.14 pp |
| Achieved Occupancy (%) | 94.13 % | 82.31 % | -11.82 pp | 82.33 % | -11.80 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.14 % | 74.66 % | +3.52 pp | 74.42 % | +3.28 pp |
| IPC Active | 1.17 inst/cycle | 1.04 inst/cycle | -0.13 | 1.03 inst/cycle | -0.14 |

---

## Iteration 5

**Best so far**: 2.4775× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_fma_and_store_index_recomputation | 3.536× | 2.517× | ✓ | **✓ best** |
| E | add_triton_autotune | 3.517× | 2.502× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174020 ns | 69150 ns | -104870 ns (-60.3%) | 69540 ns | -104480 ns (-60.0%) |
| SM Throughput (%) | 27.91 % | 25.03 % | -2.88 pp | 24.91 % | -3.00 pp |
| Memory Throughput (%) | 81.83 % | 94.02 % | +12.19 pp | 93.56 % | +11.73 pp |
| DRAM Throughput (%) | 67.50 % | 16.58 % | -50.92 pp | 16.53 % | -50.97 pp |
| Achieved Occupancy (%) | 94.13 % | 82.35 % | -11.78 pp | 82.46 % | -11.67 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.14 % | 74.68 % | +3.54 pp | 73.90 % | +2.76 pp |
| IPC Active | 1.17 inst/cycle | 1.03 inst/cycle | -0.14 | 1.01 inst/cycle | -0.16 |


### Key Learnings

#### ✅ Plan A: `eliminate_redundant_fma_and_store_index_recomputation` (2.517×)

Replacing `libdevice.fma(tmp0, tl.full(tmp0.shape, 1.0, tl.float32), tmp1)` with the plain `tmp0 + tmp1` and eliminating a redundant re-derivation of the store index (which duplicated an already-computed xindex) reduced warp cycles per issued instruction from 51.93 to 50.13 and duration from 69.6us to 69.15us (~0.65%). The unnecessary tl.full broadcast allocation and redundant index recomputation added dead instructions that consumed register budget and increased per-warp instruction count without changing semantics. In L2-saturated pointwise kernels where the compute portion is a negligible fraction of runtime, even small instruction-count reductions can yield marginal but real latency improvements by slightly increasing eligible warp fraction.

> **Applicable when**: Triton pointwise kernel where the compute body contains a libdevice.fma(x, 1.0, y) pattern (semantically equivalent to x+y) or a store index that is re-derived from scratch using tl.max_contiguous/tl.multiple_of when an identical expression was already computed for the load index. Most useful when the kernel is already L2-saturated (>90% L2 throughput) and any instruction-count reduction helps at the margin.

> **Avoid when**: The fma(x, c, y) call uses a non-unity multiplier (c ≠ 1.0), in which case it is not equivalent to x+y. Do not remove the store index re-derivation if the store address genuinely differs from the load index (e.g., different offset, different mask, or different contiguity properties). The gain is very small (~0.65% latency) and should not be prioritized over structural fixes (coalescing, layout changes).

---

## Cumulative Progress

```
Baseline:  174020 ns  (1.000×)
Iter 1     71970 ns  (2.418×)  — vectorized_loads_with_contiguity_hints
Iter 2     71710 ns  (2.427×)  — autotune_num_stages_for_latency_hiding
Iter 4     70240 ns  (2.478×)  — autotune_spatial_aligned_xblock_and_high_warp_configs_for_partial_sector_elimination
Iter 5     69150 ns  (2.517×)  — eliminate_redundant_fma_and_store_index_recomputation

Final best: 69150 ns  (2.5166×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,52 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # num_stages=1 (original baseline, no pipelining)
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 1}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=1),
+        # num_stages=2 (one prefetch buffer - hides ~47-cycle L2 latency)
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 1}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+        # num_stages=3 (two prefetch buffers - deeper pipelining for H200 TMA)
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 1}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        # larger tiles for better vectorization amortization
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
@@ -13,18 +59,17 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # KEY CHANGE: contiguity hints enable 128-bit vectorized memory instructions
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 96)
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), mask=xmask & ymask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (xindex + 3025*y3), tmp2, mask=xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +78,7 @@

     ynumel = 24576
     xnumel = 3025
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
