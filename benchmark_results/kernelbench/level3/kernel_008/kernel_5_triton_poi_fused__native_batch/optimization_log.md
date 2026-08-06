# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 124900 ns — **Best latency**: 202980 ns (**0.6153×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_008/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 29.70
- **Registers/thread**: 62
- **Duration**: 124900 ns
- **SM Throughput**: 60.98 % | **Memory**: 75.60 % | **DRAM**: 61.16 %
- **Achieved Occupancy**: 47.51 %
- **IPC Active**: 2.51 inst/cycle | **No Eligible**: 37.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_register_and_cache_efficiency | — | 0.615× | ✓ | **✓ best** |
| B | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | — | 1.004× | ✓ |  |
| C | add_autotune_with_vectorized_tile_configs | — | 0.732× | ✓ |  |
| D | add_autotune_reduce_register_pressure | — | 0.620× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | add_autotune_with... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 124900 ns | 202980 ns | +78080 ns (+62.5%) | 124420 ns | -480 ns (-0.4%) | 170720 ns | +45820 ns (+36.7%) | 201540 ns | +76640 ns (+61.4%) |
| SM Throughput (%) | 60.98 % | 58.17 % | -2.81 pp | 60.69 % | -0.29 pp | 73.87 % | +12.89 pp | 58.15 % | -2.83 pp |
| Memory Throughput (%) | 75.60 % | 50.44 % | -25.16 pp | 75.83 % | +0.23 pp | 59.76 % | -15.84 pp | 50.66 % | -24.94 pp |
| DRAM Throughput (%) | 61.16 % | 37.77 % | -23.39 pp | 61.41 % | +0.25 pp | 5.38 % | -55.78 pp | 38.05 % | -23.11 pp |
| Achieved Occupancy (%) | 47.51 % | 29.87 % | -17.64 pp | 47.51 % | +0.00 pp | 29.80 % | -17.71 pp | 29.88 % | -17.63 pp |
| Registers/Thread | 62.00 register/thread | 96.00 register/thread | +34 | 62.00 register/thread | +0 | 96.00 register/thread | +34 | 96.00 register/thread | +34 |
| Waves/SM | 29.70 | 47.52 | +17.82 | 29.70 | +0.00 | 47.52 | +17.82 | 47.52 | +17.82 |
| No Eligible (%) | 37.45 % | 41.89 % | +4.44 pp | 36.79 % | -0.66 pp | 24.65 % | -12.80 pp | 41.15 % | +3.70 pp |
| IPC Active | 2.51 inst/cycle | 2.34 inst/cycle | -0.17 | 2.53 inst/cycle | +0.02 | 3.01 inst/cycle | +0.50 | 2.35 inst/cycle | -0.16 |


### Key Learnings

#### ❌ Plan A: `autotune_tile_sizes_for_register_and_cache_efficiency` (0.635×)

Attempting to restore the original fixed tile config by removing @triton.autotune still resulted in a regression (0.63x) because the code diff introduced xmask & ymask on 1D parameter loads (indexed only by y0) instead of ymask-only, and the reciprocal computation was changed from an int32-based pattern to a float literal — yet registers per thread still jumped from 62 to 96, collapsing theoretical occupancy from 50% to 31.25% and achieved occupancy from 47.5% to 29.85%. The instruction count nearly doubled (57.5M → 90.7M) and duration increased from 124us to 197us. The profiling data shows that even without autotune, changes to the kernel body (mask widening on 1D loads, arithmetic reformulation) can cause the compiler to allocate significantly more registers, making occupancy the dominant bottleneck.

> **Applicable when**: Never apply this pattern. Widening masks from ymask to xmask & ymask on loads indexed only by the y-dimension is semantically redundant and causes register pressure increases that collapse occupancy.

> **Avoid when**: Do not apply xmask to loads that are indexed solely by the y-dimension (1D channel-indexed parameters). This adds unnecessary predicate live values, inflating register count and collapsing occupancy. Even when removing autotune, any kernel body change that increases live values per thread (mask widening, arithmetic reformulation) can cause the compiler to allocate more registers and regress performance when the kernel is already register-limited (Block Limit Registers < Block Limit Warps).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_int32_reciprocal_with_float_rsqrt_to_reduce_register_pressure_and_enable_fma | — | 1.032× | ✓ |  |
| B | simplify_reciprocal_sqrt_to_reduce_register_pressure | — | 0.965× | ✓ |  |
| C | restore_fixed_tile_config_to_recover_occupancy_and_reduce_instructions | — | 0.635× | ✓ |  |
| D | reduce_register_pressure_via_reciprocal_simplification_and_mask_cleanup | — | 0.598× | ✓ |  |
| E | add_triton_autotune | — | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_int32_rec... | Δ | simplify_reciproc... | Δ | restore_fixed_til... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 124900 ns | 121060 ns | -3840 ns (-3.1%) | 129440 ns | +4540 ns (+3.6%) | 196740 ns | +71840 ns (+57.5%) | 208770 ns | +83870 ns (+67.1%) | 126560 ns | +1660 ns (+1.3%) |
| SM Throughput (%) | 60.98 % | 33.40 % | -27.58 pp | 62.18 % | +1.20 pp | 59.70 % | -1.28 pp | 56.95 % | -4.03 pp | 62.15 % | +1.17 pp |
| Memory Throughput (%) | 75.60 % | 77.91 % | +2.31 pp | 79.31 % | +3.71 pp | 52.15 % | -23.45 pp | 48.45 % | -27.15 pp | 74.62 % | -0.98 pp |
| DRAM Throughput (%) | 61.16 % | 63.00 % | +1.84 pp | 6.91 % | -54.25 pp | 39.00 % | -22.16 pp | 36.73 % | -24.43 pp | 60.38 % | -0.78 pp |
| Achieved Occupancy (%) | 47.51 % | 59.81 % | +12.30 pp | 29.34 % | -18.17 pp | 29.85 % | -17.66 pp | 29.91 % | -17.60 pp | 47.53 % | +0.02 pp |
| Registers/Thread | 62.00 register/thread | 46.00 register/thread | -16 | 96.00 register/thread | +34 | 96.00 register/thread | +34 | 96.00 register/thread | +34 | 62.00 register/thread | +0 |
| Waves/SM | 29.70 | 23.76 | -5.94 | 47.52 | +17.82 | 47.52 | +17.82 | 47.52 | +17.82 | 29.70 | +0.00 |
| No Eligible (%) | 37.45 % | 65.42 % | +27.97 pp | 36.12 % | -1.33 pp | 39.94 % | +2.49 pp | 42.58 % | +5.13 pp | 37.51 % | +0.06 pp |
| IPC Active | 2.51 inst/cycle | 1.35 inst/cycle | -1.16 | 2.55 inst/cycle | +0.04 | 2.40 inst/cycle | -0.11 | 2.30 inst/cycle | -0.21 | 2.51 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan D: `reduce_register_pressure_via_reciprocal_simplification_and_mask_cleanup` (0.598×)

Replacing the int32 reciprocal pattern (`tl.full([1,1],1,tl.int32) / x * 1.0`) with a direct float reciprocal (`1.0 / x`) and simultaneously adding xmask to previously ymask-only 1D loads did not reduce register pressure — registers per thread remained at 96 (unchanged from the prior iteration's regressed state), theoretical occupancy stayed at 31.25%, and duration increased further from ~203us to ~209us (0.60x speedup). The instruction count grew from ~91M to ~93M, indicating the autotune config changes (new YBLOCK/XBLOCK combinations) introduced additional work. The core lesson is that when the kernel is already in a register-pressure-regressed state (96 regs/thread) due to autotune config selection, source-level arithmetic simplifications alone cannot recover occupancy — the register allocator's behavior is dominated by the tile size and live-value count determined by the autotuner's selected config, not by individual expression simplifications.

> **Applicable when**: Never apply this pattern in isolation when the kernel is already register-pressure-limited due to autotune config selection. Source-level arithmetic simplifications (int32→float reciprocal) do not reliably reduce register count when the compiler's register allocation is dominated by tile size and live-value count from the autotuner.

> **Avoid when**: Do not apply float reciprocal simplification as a register-pressure fix when: (1) the kernel is already in a regressed state with 96 regs/thread from a prior autotune change; (2) the autotune config space is simultaneously being changed to include larger YBLOCK values (e.g., 16, 32, 64), as these increase live values and register pressure further; (3) adding xmask to ymask-only loads — this adds unnecessary predicate instructions and increases instruction count without benefit.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_eviction_policy_large_tensors_to_evict_first | — | 0.715× | ✓ |  |
| C | revert_1d_load_masks_to_ymask_only_to_recover_occupancy | — | 0.988× | ✓ |  |
| D | replace_sqrt_reciprocal_with_rsqrt_to_enable_fma_and_reduce_live_values | — | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_eviction_p... | Δ | revert_1d_load_ma... | Δ | replace_sqrt_reci... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 124900 ns | 174620 ns | +49720 ns (+39.8%) | 126430 ns | +1530 ns (+1.2%) | 123330 ns | -1570 ns (-1.3%) |
| SM Throughput (%) | 60.98 % | 23.09 % | -37.89 pp | 60.99 % | +0.01 pp | 39.37 % | -21.61 pp |
| Memory Throughput (%) | 75.60 % | 63.24 % | -12.36 pp | 74.57 % | -1.03 pp | 85.14 % | +9.54 pp |
| DRAM Throughput (%) | 61.16 % | 58.02 % | -3.14 pp | 60.43 % | -0.73 pp | 62.21 % | +1.05 pp |
| Achieved Occupancy (%) | 47.51 % | 60.15 % | +12.64 pp | 47.48 % | -0.03 pp | 41.45 % | -6.06 pp |
| Registers/Thread | 62.00 register/thread | 46.00 register/thread | -16 | 62.00 register/thread | +0 | 72.00 register/thread | +10 |
| Waves/SM | 29.70 | 23.76 | -5.94 | 29.70 | +0.00 | 33.94 | +4.24 |
| No Eligible (%) | 37.45 % | 76.95 % | +39.50 pp | 35.63 % | -1.82 pp | 58.96 % | +21.51 pp |
| IPC Active | 2.51 inst/cycle | 0.92 inst/cycle | -1.59 | 2.57 inst/cycle | +0.06 | 1.65 inst/cycle | -0.86 |


### Key Learnings

#### ❌ Plan A: `change_eviction_policy_large_tensors_to_evict_first` (0.715×)

Switching large 2D streaming tensor loads from eviction_policy='evict_last' to 'evict_first' in a Triton kernel caused a 28% slowdown (0.72x). Contrary to the hypothesis that evict_first would free L2 space for small channel-parameter tensors, the L2 hit rate dropped from 55.58% to 41.64% and warp cycles per issued instruction increased from 28.4 to 41.3 — indicating that the large tensors were actually benefiting from L2 residency across the 23.76 waves/SM execution, and evict_first caused more DRAM traffic. The L1/TEX hit rate was unchanged (7.08% → 7.01%), confirming the small 1D parameters were not the bottleneck and did not benefit from the freed L2 space.

> **Applicable when**: Never apply evict_first to large tensors when the kernel has many waves per SM (>10) and the L2 hit rate for those tensors is already meaningful (>40%). The large tensors may be partially reused across waves within the same SM, making L2 retention beneficial.

> **Avoid when**: Do not switch large streaming tensor loads from evict_last to evict_first when: (1) the kernel runs 20+ waves per SM, as later waves on the same SM may reuse data cached by earlier waves; (2) the L2 hit rate is already above 40% for the large tensors, indicating genuine L2 reuse; (3) the L1/TEX hit rate for small parameter tensors is already near-zero (7%), suggesting those tensors are not the cache pressure source. The evict_first hint (.cs in CUDA) is only beneficial for truly one-pass streaming access with no inter-wave reuse.

---

## Cumulative Progress

```
Baseline:  124900 ns  (1.000×)

Final best: 202980 ns  (0.6153×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 4}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 640
@@ -20,31 +33,25 @@

     y1 = yindex // 64
     y3 = yindex
     tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
+    tmp17 = tl.load(in_ptr5 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr6 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr7 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr8 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1, 1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1, 1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp18 = tmp16 - tmp17
     tmp20 = tmp19 + tmp4
-    tmp21 = tl.sqrt_rn(tmp20)
-    tmp22 = (tmp7 / tmp21)
-    tmp23 = tmp22 * tmp9
+    tmp23 = tl.math.rsqrt(tmp20)
     tmp24 = tmp18 * tmp23
     tmp26 = tmp24 * tmp25
     tmp28 = tmp26 + tmp27
@@ -52,8 +59,6 @@

     tmp30 = tl.full([1, 1], 0, tl.int32)
     tmp31 = tl.maximum(tmp30, tmp29)
     tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -70,7 +75,7 @@

     ynumel = 640
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176)
     return arg_10
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
