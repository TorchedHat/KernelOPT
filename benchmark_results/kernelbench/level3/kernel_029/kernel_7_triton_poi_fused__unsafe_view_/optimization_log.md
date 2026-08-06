# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31620 ns — **Best latency**: 31620 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_7_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (29768,1,1), Waves/SM: 14.09
- **Registers/thread**: 24
- **Duration**: 31620 ns
- **SM Throughput**: 64.93 % | **Memory**: 14.00 % | **DRAM**: 7.95 %
- **Achieved Occupancy**: 86.21 %
- **IPC Active**: 2.79 inst/cycle | **No Eligible**: 30.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_cta_overhead | 0.955× | 0.955× | ✓ |  |
| B | increase_xblock_with_autotune | 0.992× | 0.433× | ✓ |  |
| C | autotune_xblock_and_num_warps_to_reduce_scheduler_contention | 0.987× | 0.434× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_amortized_index_arithmetic | 0.955× | 0.955× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31620 ns | 33120 ns | +1500 ns (+4.7%) | 72990 ns | +41370 ns (+130.8%) | 72800 ns | +41180 ns (+130.2%) | 33120 ns | +1500 ns (+4.7%) |
| SM Throughput (%) | 64.93 % | 70.66 % | +5.73 pp | 75.52 % | +10.59 pp | 75.56 % | +10.63 pp | 70.74 % | +5.81 pp |
| Memory Throughput (%) | 14.00 % | 19.37 % | +5.37 pp | 22.70 % | +8.70 pp | 22.70 % | +8.70 pp | 19.40 % | +5.40 pp |
| DRAM Throughput (%) | 7.95 % | 7.59 % | -0.36 pp | 3.44 % | -4.51 pp | 3.45 % | -4.50 pp | 7.63 % | -0.32 pp |
| Achieved Occupancy (%) | 86.21 % | 87.45 % | +1.24 pp | 90.30 % | +4.09 pp | 90.28 % | +4.07 pp | 87.64 % | +1.43 pp |
| Registers/Thread | 24.00 register/thread | 26.00 register/thread | +2 | 30.00 register/thread | +6 | 30.00 register/thread | +6 | 26.00 register/thread | +2 |
| Waves/SM | 14.09 | 14.09 | +0.00 | 28.19 | +14.10 | 28.19 | +14.10 | 14.09 | +0.00 |
| No Eligible (%) | 30.08 % | 24.47 % | -5.61 pp | 22.26 % | -7.82 pp | 22.20 % | -7.88 pp | 24.52 % | -5.56 pp |
| IPC Active | 2.79 inst/cycle | 3.01 inst/cycle | +0.22 | 3.10 inst/cycle | +0.31 | 3.11 inst/cycle | +0.32 | 3.01 inst/cycle | +0.22 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune` (0.433×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) to a compute-bound pointwise kernel with complex multi-dimensional index arithmetic caused a 2.3x slowdown. The profiling shows executed instructions jumped from ~16M to ~43M (2.7x increase), and duration went from 31.6µs to 73.0µs, despite occupancy improving slightly (86.2% → 90.3%). The regression is caused by the autotune overhead itself (multiple kernel launches during tuning) combined with the fact that the selected larger block size (256) increased register pressure (24 → 30 registers/thread), doubled waves/SM (14 → 28), and worsened L1 hit rate (47.8% → 35.3%), indicating that the original XBLOCK=128 was already near-optimal for this kernel's access pattern. The CPIStall 'not selected' signal (31.6%) was misleading — the kernel was already well-tuned and the stall was inherent to the compute-bound ALU pipeline, not fixable by block size changes.

> **Applicable when**: Never applicable in this form. Do NOT add autotune to increase block size when: (1) the kernel is already compute-bound with ALU pipeline >60%, (2) the kernel has complex multi-dimensional index arithmetic with many integer div/mod operations, (3) the original block size already achieves >85% occupancy, (4) CPIStall 'not selected' is the dominant stall but memory throughput is low (<20%) — this pattern indicates the stall is inherent to the compute pipeline, not a scheduling inefficiency fixable by tile size.

> **Avoid when**: Kernel is compute-bound (Compute SOL >60%) with complex index arithmetic; original block size already achieves high occupancy (>85%); CPIStall 'not selected' is present alongside low memory utilization — this combination means the stall is ALU-pipeline-inherent, not a CTA scheduling problem. Adding autotune in this scenario increases instruction count 2-3x due to register spill or worse cache behavior at larger block sizes.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.054× | 0.049× | ✓ |  |
| B | persistent_tile_scheduling | 0.014× | 0.013× | ✓ |  |
| C | persistent_tile_scheduling_with_fixed_xblock | 0.976× | 0.980× | ✓ |  |
| D | persistent_tile_scheduling_to_reduce_cta_launch_overhead | 0.014× | 0.013× | ✓ |  |
| E | add_triton_autotune | 0.987× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 31620 ns | 640100 ns | +608480 ns (+1924.4%) | 2500000 ns | +2468380 ns (+7806.4%) | 32260 ns | +640 ns (+2.0%) | 2410000 ns | +2378380 ns (+7521.8%) | 31580 ns | -40 ns (-0.1%) |
| SM Throughput (%) | 64.93 % | 79.50 % | +14.57 pp | 79.82 % | +14.89 pp | 67.78 % | +2.85 pp | 80.63 % | +15.70 pp | 64.61 % | -0.32 pp |
| Memory Throughput (%) | 14.00 % | 17.89 % | +3.89 pp | 17.70 % | +3.70 pp | 13.79 % | -0.21 pp | 18.32 % | +4.32 pp | 13.94 % | -0.06 pp |
| DRAM Throughput (%) | 7.95 % | 0.46 % | -7.49 pp | 0.12 % | -7.83 pp | 7.84 % | -0.11 pp | 0.12 % | -7.83 pp | 8.00 % | +0.05 pp |
| Achieved Occupancy (%) | 86.21 % | 99.01 % | +12.80 pp | 99.59 % | +13.38 pp | 86.36 % | +0.15 pp | 99.55 % | +13.34 pp | 85.98 % | -0.23 pp |
| Registers/Thread | 24.00 register/thread | 29.00 register/thread | +5 | 29.00 register/thread | +5 | 28.00 register/thread | +4 | 28.00 register/thread | +4 | 24.00 register/thread | +0 |
| Waves/SM | 14.09 | 14.09 | +0.00 | 14.09 | +0.00 | 14.09 | +0.00 | 14.09 | +0.00 | 14.09 | +0.00 |
| No Eligible (%) | 30.08 % | 20.24 % | -9.84 pp | 20.09 % | -9.99 pp | 27.38 % | -2.70 pp | 19.30 % | -10.78 pp | 29.99 % | -0.09 pp |
| IPC Active | 2.79 inst/cycle | 3.19 inst/cycle | +0.40 | 3.20 inst/cycle | +0.41 | 2.90 inst/cycle | +0.11 | 3.23 inst/cycle | +0.44 | 2.78 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.049×)

Converting a one-CTA-per-tile pointwise kernel (29,768 CTAs) to a persistent kernel (528 CTAs, each looping over ~56 tiles) caused a 20x slowdown. Despite L2 hit rate improving dramatically (59.67% → 97.82%) and occupancy rising to 99%, executed instructions exploded from 18M to 403M (21.8x increase), and duration jumped from 33µs to 640µs. The root cause is that the while-loop construct in Triton prevents the compiler from statically unrolling or pipelining the tile loop body, and the complex multi-dimensional index arithmetic (7+ integer div/mod operations per element) is re-executed serially per tile iteration rather than being vectorized across tiles. The grid launch parameter was also left unchanged at 29,768 (the original one-tile-per-CTA grid) rather than being reduced to 528, meaning the persistent loop ran inside a full-sized grid — compounding the instruction count explosion by ~56x. This anti-pattern is catastrophic for compute-bound kernels with complex index arithmetic.

> **Applicable when**: Never applicable for compute-bound pointwise kernels with complex multi-dimensional index arithmetic. Persistent tile scheduling may help memory-bound kernels where L2 reuse is the dominant bottleneck and the per-tile compute is minimal (e.g., simple elementwise ops with no div/mod index arithmetic).

> **Avoid when**: Kernel is compute-bound (Compute SOL >60%) with complex index arithmetic (many integer div/mod operations per element). The while-loop in Triton's persistent kernel pattern prevents compiler optimization of the loop body, causing serial re-execution of all index arithmetic per tile. Also avoid when the grid launch parameter is not simultaneously reduced to match the persistent CTA count — leaving the grid at the original size while adding a tile loop multiplies instruction count by the number of tiles per CTA.

#### ❌ Plan B: `persistent_tile_scheduling` (0.049×)

Converting a one-CTA-per-tile pointwise kernel (29,768 CTAs) to a persistent kernel (528 CTAs, each looping over ~56 tiles) caused a 20x slowdown. Despite L2 hit rate improving dramatically (59.67% → 97.82%) and occupancy rising to 99%, executed instructions exploded from 18M to 403M (21.8x increase), and duration jumped from 33µs to 640µs. The root cause is that the while-loop construct in Triton prevents the compiler from statically unrolling or pipelining the tile loop body, and the complex multi-dimensional index arithmetic (7+ integer div/mod operations per element) is re-executed serially per tile iteration rather than being vectorized across tiles. The grid launch parameter was also left unchanged at 29,768 (the original one-tile-per-CTA grid) rather than being reduced to 528, meaning the persistent loop ran inside a full-sized grid — compounding the instruction count explosion by ~56x. This anti-pattern is catastrophic for compute-bound kernels with complex index arithmetic.

> **Applicable when**: Never applicable for compute-bound pointwise kernels with complex multi-dimensional index arithmetic. Persistent tile scheduling may help memory-bound kernels where L2 reuse is the dominant bottleneck and the per-tile compute is minimal (e.g., simple elementwise ops with no div/mod index arithmetic).

> **Avoid when**: Kernel is compute-bound (Compute SOL >60%) with complex index arithmetic (many integer div/mod operations per element). The while-loop in Triton's persistent kernel pattern prevents compiler optimization of the loop body, causing serial re-execution of all index arithmetic per tile. Also avoid when the grid launch parameter is not simultaneously reduced to match the persistent CTA count — leaving the grid at the original size while adding a tile loop multiplies instruction count by the number of tiles per CTA.

---

## Cumulative Progress

```
Baseline:  31620 ns  (1.000×)

Final best: 31620 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=16),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3810240
@@ -26,25 +39,23 @@

     tmp8 = tmp2 & tmp4
     tmp9 = tmp8 & tmp6
     tmp10 = tmp9 & tmp7
-    tmp11 = tl.load(in_ptr0 + ((-21888) + x0 + 32*(x1 // 49) + 96*((((x1 % 49)) % 7)) + 672*((x2 % 9)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 9) % 9)) + 301056*(x2 // 81)), tmp10 & xmask, other=0.0)
-    tmp12 = tl.load(in_ptr1 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp11 = tl.load(in_ptr0 + ((-21888) + x0 + 32*(x1 // 49) + 96*((((x1 % 49)) % 7)) + 672*((x2 % 9)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 9) % 9)) + 301056*(x2 // 81)), mask=tmp10 & xmask, other=0.0)
+    tmp12 = tl.load(in_ptr1 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp13 = tmp11 - tmp12
-    tmp14 = tl.load(in_ptr2 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp14 = tl.load(in_ptr2 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp15 = tl.full([1], 96.0, tl.float32)
     tmp16 = (tmp14 / tmp15)
     tmp17 = tl.full([1], 1e-05, tl.float32)
     tmp18 = tmp16 + tmp17
     tmp19 = libdevice.rsqrt(tmp18)
     tmp20 = tmp13 * tmp19
-    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp22 = tmp20 * tmp21
-    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
     tmp24 = tmp22 + tmp23
     tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
     tmp26 = tl.where(tmp10, tmp24, tmp25)
-    tl.store(out_ptr0 + (x3), tmp26, xmask)
-
-
+    tl.store(out_ptr0 + (x3), tmp26, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -55,7 +66,7 @@

     arg_5 = inputs[5]
     xnumel = 3810240
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3810240, XBLOCK=128)
+    triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3810240)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
