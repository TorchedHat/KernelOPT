# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13380 ns — **Best latency**: 13180 ns (**1.0152×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`streaming_store_cache_modifier_for_l2_compression`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_19_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (11760,1,1), Waves/SM: 5.57
- **Registers/thread**: 28
- **Duration**: 13380 ns
- **SM Throughput**: 66.59 % | **Memory**: 27.55 % | **DRAM**: 27.55 %
- **Achieved Occupancy**: 82.63 %
- **IPC Active**: 3.12 inst/cycle | **No Eligible**: 20.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_cache_reuse_and_compute_efficiency | 0.988× | 1.003× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_fusion | 0.667× | 0.629× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads | 0.668× | 0.635× | ✓ |  |
| D | autotune_xblock_tile_size_for_cache_reuse_and_scheduler_efficiency | 1.000× | 0.367× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13380 ns | 13340 ns | -40 ns (-0.3%) | 21280 ns | +7900 ns (+59.0%) | 21060 ns | +7680 ns (+57.4%) | 36450 ns | +23070 ns (+172.4%) |
| SM Throughput (%) | 66.59 % | 66.38 % | -0.21 pp | 68.23 % | +1.64 pp | 69.05 % | +2.46 pp | 76.54 % | +9.95 pp |
| Memory Throughput (%) | 27.55 % | 27.69 % | +0.14 pp | 19.10 % | -8.45 pp | 19.42 % | -8.13 pp | 18.26 % | -9.29 pp |
| DRAM Throughput (%) | 27.55 % | 27.69 % | +0.14 pp | 17.32 % | -10.23 pp | 17.49 % | -10.06 pp | 10.11 % | -17.44 pp |
| Achieved Occupancy (%) | 82.63 % | 81.58 % | -1.05 pp | 63.74 % | -18.89 pp | 63.41 % | -19.22 pp | 64.58 % | -18.05 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 36.00 register/thread | +8 | 36.00 register/thread | +8 | 36.00 register/thread | +8 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 7.42 | +1.85 | 7.42 | +1.85 | 14.85 | +9.28 |
| No Eligible (%) | 20.09 % | 19.87 % | -0.22 pp | 23.61 % | +3.52 pp | 23.43 % | +3.34 pp | 18.68 % | -1.41 pp |
| IPC Active | 3.12 inst/cycle | 3.15 inst/cycle | +0.03 | 3.04 inst/cycle | -0.08 | 3.04 inst/cycle | -0.08 | 3.24 inst/cycle | +0.12 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_for_pointwise_fusion` (0.629×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (28→36), which reduced theoretical occupancy from 100% to 75% and achieved occupancy from 82.6% to 63.7%. The larger tile sizes also increased total executed instructions by ~66% (6.8M→11.4M), likely due to increased loop overhead and predicated work at tile boundaries, resulting in a 59% longer kernel duration (13.4→21.3 µs). The autotune overhead itself (extra waves: 5.57→7.42) further compounded the regression. When a kernel is already register-limited at the baseline block size, increasing tile size will push register usage over the occupancy cliff, causing net slowdown despite any ILP gains.

> **Applicable when**: Kernel has very low register pressure at baseline (e.g., <20 registers/thread) with significant headroom before the occupancy cliff, AND the kernel is clearly memory-bound with large reuse potential from wider tiles.

> **Avoid when**: Kernel is already compute-bound (Compute SOL >> Memory SOL); baseline registers_per_thread is moderate-to-high (≥24) such that larger tiles will push over the register-limited occupancy boundary; theoretical occupancy is already at 100% with the current block size; total instruction count is expected to grow significantly with larger tiles.

#### ❌ Plan C: `increase_xblock_with_autotune_for_vectorized_loads` (0.635×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (28→36), which reduced Block Limit Registers from 16 to 12 and dropped theoretical occupancy from 100% to 75% and achieved occupancy from 82.6% to 63.4%. The increased register pressure also caused a 66% increase in executed instructions (6.8M→11.4M), more than offsetting any benefit from reduced grid size or wider memory transactions. The kernel duration increased from 13.4µs to 21.1µs, a 57% regression.

> **Applicable when**: Never applicable as described: adding autotune with larger XBLOCK to a kernel that is already register-constrained (registers_per_thread near 28–32, Block Limit Registers already at 16) will increase register pressure and reduce occupancy.

> **Avoid when**: Do not increase XBLOCK via autotune when: (1) the kernel already uses close to the register budget (≥24 registers/thread), (2) theoretical occupancy is already at or near 100%, (3) the kernel is compute-bound rather than memory-latency-bound. Larger XBLOCK forces the compiler to keep more live values simultaneously, increasing register count and reducing Block Limit Registers, which collapses occupancy and increases total instruction count.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_fusion_for_non_fused_fp32_instructions | 0.988× | 1.003× | ✓ |  |
| B | remove_autotune_to_restore_register_efficient_baseline | 0.992× | 1.015× | ✓ |  |
| C | reduce_register_pressure_via_smaller_xblock_and_fewer_warps | 1.116× | 1.015× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_fusi... | Δ | remove_autotune_t... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13380 ns | 13340 ns | -40 ns (-0.3%) | 13180 ns | -200 ns (-1.5%) | 13180 ns | -200 ns (-1.5%) | 13410 ns | +30 ns (+0.2%) |
| SM Throughput (%) | 66.59 % | 67.17 % | +0.58 pp | 66.42 % | -0.17 pp | 66.94 % | +0.35 pp | 66.96 % | +0.37 pp |
| Memory Throughput (%) | 27.55 % | 27.63 % | +0.08 pp | 27.97 % | +0.42 pp | 28.03 % | +0.48 pp | 27.54 % | -0.01 pp |
| DRAM Throughput (%) | 27.55 % | 27.63 % | +0.08 pp | 27.97 % | +0.42 pp | 28.03 % | +0.48 pp | 27.54 % | -0.01 pp |
| Achieved Occupancy (%) | 82.63 % | 82.53 % | -0.10 pp | 82.79 % | +0.16 pp | 82.70 % | +0.07 pp | 83.19 % | +0.56 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 20.09 % | 19.72 % | -0.37 pp | 19.89 % | -0.20 pp | 19.86 % | -0.23 pp | 20.17 % | +0.08 pp |
| IPC Active | 3.12 inst/cycle | 3.18 inst/cycle | +0.06 | 3.14 inst/cycle | +0.02 | 3.19 inst/cycle | +0.07 | 3.18 inst/cycle | +0.06 |

---

## Iteration 3

**Best so far**: 1.0152× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_reciprocal_sqrt_with_libdevice_rsqrt_for_fma_fusion | 1.018× | 1.192× | ✓ |  |
| B | replace_sqrt_div_with_rsqrt_for_fma_fusion | 1.109× | 1.184× | ✓ |  |
| C | change_eviction_policy_streaming_loads_to_evict_first | 1.015× | 1.003× | ✓ |  |
| D | replace_sqrt_reciprocal_chain_with_rsqrt_for_fma_fusion | 1.019× | 1.212× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_re... | Δ | replace_sqrt_div_... | Δ | change_eviction_p... | Δ | replace_sqrt_reci... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13380 ns | 11230 ns | -2150 ns (-16.1%) | 11300 ns | -2080 ns (-15.5%) | 13340 ns | -40 ns (-0.3%) | 11040 ns | -2340 ns (-17.5%) |
| SM Throughput (%) | 66.59 % | 61.00 % | -5.59 pp | 61.51 % | -5.08 pp | 67.29 % | +0.70 pp | 61.19 % | -5.40 pp |
| Memory Throughput (%) | 27.55 % | 32.82 % | +5.27 pp | 32.62 % | +5.07 pp | 27.68 % | +0.13 pp | 33.43 % | +5.88 pp |
| DRAM Throughput (%) | 27.55 % | 32.82 % | +5.27 pp | 32.62 % | +5.07 pp | 27.68 % | +0.13 pp | 33.43 % | +5.88 pp |
| Achieved Occupancy (%) | 82.63 % | 76.83 % | -5.80 pp | 76.41 % | -6.22 pp | 82.36 % | -0.27 pp | 75.68 % | -6.95 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 28.00 register/thread | +0 | 30.00 register/thread | +2 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 20.09 % | 23.10 % | +3.01 pp | 23.21 % | +3.12 pp | 19.83 % | -0.26 pp | 23.43 % | +3.34 pp |
| IPC Active | 3.12 inst/cycle | 3.01 inst/cycle | -0.11 | 3.03 inst/cycle | -0.09 | 3.12 inst/cycle | +0.00 | 3.02 inst/cycle | -0.10 |


### Key Learnings

#### ✅ Plan A: `replace_manual_reciprocal_sqrt_with_libdevice_rsqrt_for_fma_fusion` (1.192×)

A 5-instruction manual reciprocal-sqrt sequence (sqrt_rn + int32 fill + int-div + float fill + fmul) prevents the compiler from emitting a single MUFU.RSQ hardware instruction and fusing surrounding multiply-adds into FFMA pairs. Replacing this chain with libdevice.rsqrt() collapses it to one instruction, reducing total executed instructions by ~22% (6.82M to 5.32M), eliminating divergent branches (89 avg → 0), and improving average active threads per warp from 29.8 to 32.0 (full warp utilization). The net effect is a 14.8% reduction in kernel duration (13.18 µs → 11.23 µs) at the cost of a minor register increase (28 → 30 regs/thread) that does not affect theoretical occupancy.

> **Applicable when**: NCU FPInstructions rule is triggered with a high ratio of non-fused to fused FP32 instructions (e.g., >4:1); kernel contains a manual reciprocal-sqrt pattern using tl.sqrt_rn + integer division + float cast; kernel is compute-bound (Compute SOL >> Memory SOL); avg active threads per warp is below 32 (indicating predicated-off threads from divergent branches in the manual sequence).

> **Avoid when**: The input to rsqrt may be zero or negative (no epsilon guard); when numerical precision requirements are stricter than 1 ULP (libdevice.rsqrt may differ by up to 1 ULP from the multi-step path); when the kernel is memory-bound rather than compute-bound (FP instruction reduction will not be the binding bottleneck).

#### ✅ Plan B: `replace_sqrt_div_with_rsqrt_for_fma_fusion` (1.184×)

When NCU's FPInstructions rule reports a high ratio of non-fused to fused FP32 instructions (>4:1), a common root cause is a two-instruction sqrt+reciprocal-division sequence (e.g., `1/sqrt(x)`) that prevents the compiler from emitting a single MUFU.RSQRT and blocks FMA fusion on downstream multiply chains. Replacing this sequence with `libdevice.rsqrt(x)` in Triton collapses the 5-instruction sequence into one hardware instruction and allows the compiler to fuse downstream multiplies into FMAs, reducing total executed instructions by ~22% (6.82M to 5.32M) and kernel duration by ~14% (13.18us to 11.3us). This change is safe for batch norm inference where variance+eps is always positive, and does not increase register pressure beyond the occupancy cliff.

> **Applicable when**: NCU FPInstructions rule is triggered with non-fused:fused FP32 ratio >3:1; kernel contains a tl.sqrt_rn + integer-reciprocal-division sequence computing 1/sqrt(x); the kernel is compute-bound (Compute SOL >> Memory SOL); the input to sqrt is guaranteed positive (e.g., variance+eps in batch norm); libdevice is already imported or can be imported without side effects.

> **Avoid when**: Input to sqrt may be zero or negative (rsqrt is undefined/NaN for non-positive inputs); when the change would push registers per thread over the occupancy cliff (check Block Limit Registers before and after); when the kernel is memory-bound rather than compute-bound (FMA fusion will not help if the bottleneck is DRAM bandwidth).

#### ✅ Plan D: `replace_sqrt_reciprocal_chain_with_rsqrt_for_fma_fusion` (1.212×)

When NCU's FPInstructions rule flags a high ratio of non-fused to fused FP32 instructions (e.g., 4:1), a common cause is a batch-norm normalization path that computes tl.sqrt_rn(var+eps) followed by int(1)/sqrt * 1.0 — a 3-instruction sequence (sqrt, integer-reciprocal, dead multiply) that the compiler cannot fuse into FMA. Replacing this chain with tl.math.rsqrt(var+eps) collapses sqrt+reciprocal into a single MUFU.RSQ hardware instruction and enables subsequent multiplies to fuse as FMAs, reducing total executed instructions by ~22% (6.8M to 5.3M) and kernel duration by ~16% (13.2us to 11.0us). The change also eliminates warp divergence (branch efficiency dropped from 83% to 0% divergent branches, avg active threads per warp increased from 29.8 to 32.0), indicating the original integer-division path introduced predicated/divergent code paths.

> **Applicable when**: NCU FPInstructions rule is triggered with a high non-fused to fused FP32 ratio (>3:1); kernel contains batch-norm normalization using tl.sqrt_rn followed by integer reciprocal (int(1)/sqrt) or equivalent multi-step reciprocal pattern; kernel is compute-bound (Compute SOL > Memory SOL); var+eps is guaranteed positive (epsilon > 0 ensures this for batch norm).

> **Avoid when**: The input to rsqrt could be zero or negative (rsqrt is undefined/inf for non-positive inputs); when numerical precision requirements are stricter than float32 rsqrt (tl.math.rsqrt uses hardware MUFU.RSQ which has ~23-bit precision, same as float32 sqrt+rcp); when the non-fused instruction count is already low and FPInstructions rule is not triggered.

---

## Iteration 4

**Best so far**: 1.0152× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_store_cache_modifier_for_l2_compression | 1.026× | 1.015× | ✓ |  |
| B | reduce_num_warps_for_l1_cache_coherence_and_scheduler_efficiency | 1.019× | 1.167× | ✓ |  |
| C | reorder_index_decomposition_for_coalesced_l2_access | 1.102× | 1.178× | ✓ |  |
| D | fix_autotune_grid_and_tune_num_warps_for_occupancy | 1.013× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_store_c... | Δ | reduce_num_warps_... | Δ | reorder_index_dec... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13380 ns | 13180 ns | -200 ns (-1.5%) | 11460 ns | -1920 ns (-14.3%) | 11360 ns | -2020 ns (-15.1%) | 13280 ns | -100 ns (-0.7%) |
| SM Throughput (%) | 66.59 % | 66.64 % | +0.05 pp | 60.84 % | -5.75 pp | 61.45 % | -5.14 pp | 66.90 % | +0.31 pp |
| Memory Throughput (%) | 27.55 % | 27.96 % | +0.41 pp | 32.20 % | +4.65 pp | 32.51 % | +4.96 pp | 27.83 % | +0.28 pp |
| DRAM Throughput (%) | 27.55 % | 27.96 % | +0.41 pp | 32.20 % | +4.65 pp | 32.51 % | +4.96 pp | 27.83 % | +0.28 pp |
| Achieved Occupancy (%) | 82.63 % | 81.88 % | -0.75 pp | 77.55 % | -5.08 pp | 76.89 % | -5.74 pp | 82.36 % | -0.27 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 28.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 20.09 % | 19.87 % | -0.22 pp | 25.09 % | +5.00 pp | 23.45 % | +3.36 pp | 19.81 % | -0.28 pp |
| IPC Active | 3.12 inst/cycle | 3.18 inst/cycle | +0.06 | 3.04 inst/cycle | -0.08 | 3.00 inst/cycle | -0.12 | 3.17 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  13380 ns  (1.000×)
Iter 2     13180 ns  (1.015×)  — reduce_register_pressure_via_smaller_xblock_and_fewer_warps

Final best: 13180 ns  (1.0152×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1505280
@@ -16,21 +26,21 @@

     x2 = xindex // 5376
     x3 = xindex // 192
     x4 = xindex
-    tmp27 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp29 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
-    tmp38 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
-    tmp40 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp29 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp38 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp40 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 128, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -40,29 +50,23 @@

     tmp17 = tl.full([1], 160, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 192, tl.int64)
     tmp23 = tmp0 < tmp22
-    tmp24 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp24 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), mask=tmp21 & xmask, eviction_policy='evict_last', other=0.0)
     tmp25 = tl.where(tmp19, tmp20, tmp24)
     tmp26 = tl.where(tmp4, tmp15, tmp25)
     tmp28 = tmp26 - tmp27
     tmp30 = tl.full([1], 1e-05, tl.float32)
     tmp31 = tmp29 + tmp30
-    tmp32 = tl.sqrt_rn(tmp31)
-    tmp33 = tl.full([1], 1, tl.int32)
-    tmp34 = (tmp33 / tmp32)
-    tmp35 = tl.full([1], 1.0, tl.float32)
-    tmp36 = tmp34 * tmp35
+    tmp36 = tl.math.rsqrt(tmp31)
     tmp37 = tmp28 * tmp36
     tmp39 = tmp37 * tmp38
     tmp41 = tmp39 + tmp40
     tmp42 = tl.full([1], 0, tl.int32)
     tmp43 = tl.maximum(tmp42, tmp41)
-    tl.store(in_out_ptr0 + (x4), tmp43, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp43, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +79,7 @@

     arg_7 = inputs[7]
     xnumel = 1505280
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 1505280, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 1505280)
     return arg_7
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
