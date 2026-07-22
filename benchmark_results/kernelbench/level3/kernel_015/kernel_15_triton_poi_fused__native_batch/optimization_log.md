# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 53540 ns — **Best latency**: 42430 ns (**1.2618×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restore_small_xblock_with_num_stages1_and_fma_correct_xmask`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_15_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62720,1,1), Waves/SM: 29.70
- **Registers/thread**: 32
- **Duration**: 53540 ns
- **SM Throughput**: 77.91 % | **Memory**: 32.72 % | **DRAM**: 17.23 %
- **Achieved Occupancy**: 85.11 %
- **IPC Active**: 3.24 inst/cycle | **No Eligible**: 18.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.595× | 0.580× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_autotune | 1.000× | 0.400× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53540 ns | 92380 ns | +38840 ns (+72.5%) | 133760 ns | +80220 ns (+149.8%) |
| SM Throughput (%) | 77.91 % | 84.57 % | +6.66 pp | 75.48 % | -2.43 pp |
| Memory Throughput (%) | 32.72 % | 23.34 % | -9.38 pp | 27.74 % | -4.98 pp |
| DRAM Throughput (%) | 17.23 % | 9.83 % | -7.40 pp | 6.73 % | -10.50 pp |
| Achieved Occupancy (%) | 85.11 % | 66.81 % | -18.30 pp | 60.66 % | -24.45 pp |
| Registers/Thread | 32.00 register/thread | 38.00 register/thread | +6 | 40.00 register/thread | +8 |
| Waves/SM | 29.70 | 39.60 | +9.90 | 79.19 | +49.49 |
| No Eligible (%) | 18.78 % | 13.59 % | -5.19 pp | 23.27 % | +4.49 pp |
| IPC Active | 3.24 inst/cycle | 3.45 inst/cycle | +0.21 | 3.06 inst/cycle | -0.18 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_remove_autotune_and_use_rsqrt_fma | 1.024× | 1.100× | ✓ |  |
| B | remove_autotune_restore_xblock128_and_replace_sqrt_with_rsqrt | 1.017× | 1.250× | ✓ |  |
| C | cap_registers_with_maxnreg_to_restore_occupancy | 1.020× | 0.392× | ✓ |  |
| D | reduce_register_pressure_via_smaller_xblock_and_num_stages | 1.031× | 1.101× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | remove_autotune_r... | Δ | cap_registers_wit... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53540 ns | 48670 ns | -4870 ns (-9.1%) | 42820 ns | -10720 ns (-20.0%) | 136510 ns | +82970 ns (+155.0%) | 48640 ns | -4900 ns (-9.2%) |
| SM Throughput (%) | 77.91 % | 78.15 % | +0.24 pp | 72.41 % | -5.50 pp | 80.39 % | +2.48 pp | 78.18 % | +0.27 pp |
| Memory Throughput (%) | 32.72 % | 35.87 % | +3.15 pp | 41.30 % | +8.58 pp | 39.75 % | +7.03 pp | 35.88 % | +3.16 pp |
| DRAM Throughput (%) | 17.23 % | 18.96 % | +1.73 pp | 21.50 % | +4.27 pp | 7.04 % | -10.19 pp | 19.01 % | +1.78 pp |
| Achieved Occupancy (%) | 85.11 % | 82.86 % | -2.25 pp | 78.06 % | -7.05 pp | 86.36 % | +1.25 pp | 82.32 % | -2.79 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 26.00 register/thread | -6 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 59.39 | +29.69 | 29.70 | +0.00 |
| No Eligible (%) | 18.78 % | 18.11 % | -0.67 pp | 23.07 % | +4.29 pp | 18.17 % | -0.61 pp | 18.32 % | -0.46 pp |
| IPC Active | 3.24 inst/cycle | 3.26 inst/cycle | +0.02 | 3.06 inst/cycle | -0.18 | 3.27 inst/cycle | +0.03 | 3.26 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.1007× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_and_xblock_to_256_for_fma_and_scheduler_efficiency | 1.025× | 1.098× | ✓ |  |
| B | eliminate_num_stages2_configs_to_reduce_compiler_overhead_and_improve_fma_emission | 1.031× | 1.093× | ✓ |  |
| C | inline_fma_for_batchnorm_arithmetic_chain | 1.025× | 1.088× | ✓ |  |
| D | remove_maxnreg_to_eliminate_register_spill | 1.020× | 0.400× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | eliminate_num_sta... | Δ | inline_fma_for_ba... | Δ | remove_maxnreg_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53540 ns | 48770 ns | -4770 ns (-8.9%) | 48960 ns | -4580 ns (-8.6%) | 49220 ns | -4320 ns (-8.1%) | 133730 ns | +80190 ns (+149.8%) |
| SM Throughput (%) | 77.91 % | 77.94 % | +0.03 pp | 78.10 % | +0.19 pp | 78.11 % | +0.20 pp | 75.50 % | -2.41 pp |
| Memory Throughput (%) | 32.72 % | 35.77 % | +3.05 pp | 35.85 % | +3.13 pp | 35.85 % | +3.13 pp | 27.74 % | -4.98 pp |
| DRAM Throughput (%) | 17.23 % | 18.95 % | +1.72 pp | 18.90 % | +1.67 pp | 18.80 % | +1.57 pp | 6.76 % | -10.47 pp |
| Achieved Occupancy (%) | 85.11 % | 83.13 % | -1.98 pp | 83.32 % | -1.79 pp | 83.29 % | -1.82 pp | 60.73 % | -24.38 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 | 26.00 register/thread | -6 | 26.00 register/thread | -6 | 40.00 register/thread | +8 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 79.19 | +49.49 |
| No Eligible (%) | 18.78 % | 18.47 % | -0.31 pp | 18.13 % | -0.65 pp | 18.08 % | -0.70 pp | 23.28 % | +4.50 pp |
| IPC Active | 3.24 inst/cycle | 3.27 inst/cycle | +0.03 | 3.26 inst/cycle | +0.02 | 3.26 inst/cycle | +0.02 | 3.06 inst/cycle | -0.18 |


### Key Learnings

#### ✅ Plan C: `inline_fma_for_batchnorm_arithmetic_chain` (1.088×)

Replacing a two-step multiply-add (tmp53 = tmp51 * tmp52; tmp55 = tmp53 + tmp54) with libdevice.fma(tmp51, tmp52, tmp54) in a Triton batchnorm affine transform reduced registers per thread from 32 to 26 and improved achieved occupancy from 78.1% to 83.3%, but simultaneously increased executed instructions from 24.1M to 29.9M (+24%) and elapsed cycles from 64K to 73K, causing a net slowdown (duration 42.82 µs → 49.22 µs). The instruction count increase suggests the compiler expanded the fma call into additional bookkeeping or the xmask change from tl.full([XBLOCK], True) to xindex < xnumel added predication overhead across all loads and stores. The FPInstructions NCU rule remained present after the change, indicating the fused instruction count did not improve as expected.

> **Applicable when**: Never apply as described: replacing a two-step mul+add with libdevice.fma while simultaneously changing xmask from tl.full([XBLOCK], True) to xindex < xnumel causes instruction count to balloon due to added predication overhead on all memory operations, negating any FMA fusion benefit.

> **Avoid when**: When the kernel already uses xmask = tl.full([XBLOCK], True, tl.int1)[:] (i.e., xnumel is divisible by XBLOCK and no boundary masking is needed), do NOT change the xmask pattern while applying libdevice.fma. The predication overhead from switching to xindex < xnumel adds masked instructions to every load and store, increasing total instruction count by ~24% and more than offsetting any FMA fusion savings. Also do not apply if the FPInstructions NCU rule's non-fused ops originate from branches other than the targeted mul+add pair.

---

## Iteration 4

**Best so far**: 1.1007× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | xblock256_with_full_mask_no_predication_overhead | 1.020× | 1.264× | ✓ |  |
| B | restore_full_mask_to_eliminate_predication_overhead | 1.027× | 1.264× | ✓ |  |
| C | change_large_input_eviction_policy_to_evict_first_for_streaming_loads | 1.026× | 1.101× | ✓ |  |
| D | restore_small_xblock_with_num_stages1_and_fma_correct_xmask | 1.034× | 1.262× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | xblock256_with_fu... | Δ | restore_full_mask... | Δ | change_large_inpu... | Δ | restore_small_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53540 ns | 42340 ns | -11200 ns (-20.9%) | 42370 ns | -11170 ns (-20.9%) | 48610 ns | -4930 ns (-9.2%) | 42430 ns | -11110 ns (-20.8%) |
| SM Throughput (%) | 77.91 % | 72.06 % | -5.85 pp | 72.82 % | -5.09 pp | 78.41 % | +0.50 pp | 72.83 % | -5.08 pp |
| Memory Throughput (%) | 32.72 % | 41.09 % | +8.37 pp | 41.53 % | +8.81 pp | 35.97 % | +3.25 pp | 41.53 % | +8.81 pp |
| DRAM Throughput (%) | 17.23 % | 21.84 % | +4.61 pp | 21.82 % | +4.59 pp | 16.28 % | -0.95 pp | 21.73 % | +4.50 pp |
| Achieved Occupancy (%) | 85.11 % | 77.49 % | -7.62 pp | 76.26 % | -8.85 pp | 83.04 % | -2.07 pp | 79.34 % | -5.77 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 26.00 register/thread | -6 | 32.00 register/thread | +0 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 18.78 % | 23.72 % | +4.94 pp | 23.01 % | +4.23 pp | 18.00 % | -0.78 pp | 23.20 % | +4.42 pp |
| IPC Active | 3.24 inst/cycle | 3.06 inst/cycle | -0.18 | 3.06 inst/cycle | -0.18 | 3.27 inst/cycle | +0.03 | 3.07 inst/cycle | -0.17 |


### Key Learnings

#### ✅ Plan B: `restore_full_mask_to_eliminate_predication_overhead` (1.264×)

When xnumel is exactly divisible by XBLOCK, replacing the runtime boundary check `xmask = xindex < xnumel` with the compile-time constant `xmask = tl.full([XBLOCK], True, tl.int1)[:]` eliminates predication from every load and store, reducing executed instructions by ~19% (29.9M → 24.1M) and kernel duration by ~13% (48.77 µs → 42.37 µs). The FPInstructions NCU rule flags this pattern because predicated instructions inflate non-fused FP32 instruction counts; removing the predication allows the compiler to emit unconditional PTX load/store instructions. Note that registers per thread increased from 26 to 32 (a side effect of the autotune config change), but the instruction count reduction dominated and produced a net speedup.

> **Applicable when**: xnumel is exactly divisible by all XBLOCK values in the autotune configs (verified: xnumel % XBLOCK == 0 for every config). The FPInstructions NCU rule is triggered with elevated non-fused instruction counts. The kernel uses `xmask = xindex < xnumel` and passes xmask (or compound predicates like tmp4 & xmask) to tl.load and tl.store mask arguments. Executed instruction count is anomalously high relative to the number of memory operations.

> **Avoid when**: xnumel is NOT exactly divisible by XBLOCK for all autotune configs — removing the boundary mask will cause out-of-bounds memory accesses and incorrect results. Do not apply if the kernel already uses tl.full([XBLOCK], True) masking. Do not apply if the instruction count inflation is caused by something other than predication (e.g., register spilling, branch divergence).

#### ✅ Plan C: `change_large_input_eviction_policy_to_evict_first_for_streaming_loads` (1.101×)

When a kernel loads large streaming tensors (each accessed once per CTA) alongside small frequently-reused parameter tensors (e.g., batchnorm weight/bias/mean/var, ~1KB total), marking the large streaming loads with eviction_policy='evict_first' (cache-streaming hint, maps to PTX .cs) prevents them from evicting the small reused tensors from L1/L2. In this case, DRAM throughput dropped from 18.95% to 16.28% and Average DRAM Active Cycles fell from 29,537 to 25,305 (~14% reduction), indicating the BN parameter tensors were better retained in cache. The overall speedup was modest (1.10x, ~0.16 µs reduction) because the kernel is primarily compute-bound (Compute SOL ~78%) and the memory improvement only partially addresses the 18% scheduler no-eligible stall.

> **Applicable when**: Kernel loads multiple large streaming tensors (each element read once per CTA, no reuse) alongside small parameter tensors (e.g., BN weight/bias/mean/var, <4KB total) that are reused across all CTAs at the same offsets. DRAM Throughput is non-trivial (>10%), L2 Hit Rate is moderate (~50%), and the kernel has many concurrent CTAs (high waves_per_sm). The small parameter tensors are being evicted from L1/L2 by the large streaming inputs.

> **Avoid when**: Kernel is purely compute-bound with negligible DRAM traffic (DRAM Throughput <5%), or all tensors are large streaming with no reused small tensors to protect. Do not apply when the large inputs are actually reused across multiple CTAs (e.g., weight matrices in GEMM). Speedup is limited when the kernel is already compute-bound (Compute SOL >75%); expect only modest gains (~5-15% DRAM reduction, <5% overall latency improvement).

---

## Iteration 5

**Best so far**: 1.2618× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_4_for_scheduler_coherence_and_l1tex_latency_hiding | 1.022× | 1.106× | ✓ |  |
| B | add_num_warps16_config_and_fma_to_reduce_scheduler_stall_and_fp_instructions | 1.030× | 1.257× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | add_num_warps16_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53540 ns | 48420 ns | -5120 ns (-9.6%) | 42590 ns | -10950 ns (-20.5%) |
| SM Throughput (%) | 77.91 % | 78.09 % | +0.18 pp | 73.00 % | -4.91 pp |
| Memory Throughput (%) | 32.72 % | 35.82 % | +3.10 pp | 41.62 % | +8.90 pp |
| DRAM Throughput (%) | 17.23 % | 16.35 % | -0.88 pp | 21.72 % | +4.49 pp |
| Achieved Occupancy (%) | 85.11 % | 83.44 % | -1.67 pp | 77.94 % | -7.17 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 | 32.00 register/thread | +0 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 18.78 % | 18.06 % | -0.72 pp | 23.74 % | +4.96 pp |
| IPC Active | 3.24 inst/cycle | 3.27 inst/cycle | +0.03 | 3.07 inst/cycle | -0.17 |

---

## Cumulative Progress

```
Baseline:  53540 ns  (1.000×)
Iter 2     48640 ns  (1.101×)  — reduce_register_pressure_via_smaller_xblock_and_num_stages
Iter 4     42430 ns  (1.262×)  — restore_small_xblock_with_num_stages1_and_fma_correct_xmask

Final best: 42430 ns  (1.2618×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
@@ -14,45 +15,45 @@

     x0 = (xindex % 256)
     x1 = xindex // 256
     x2 = xindex
-    tmp41 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
-    tmp43 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
-    tmp52 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
-    tmp54 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
+    tmp41 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp43 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp52 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp54 = tl.load(in_ptr10 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 64, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), mask=tmp4, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1], 96, tl.int64)
     tmp8 = tmp0 < tmp7
     tmp9 = tmp6 & tmp8
-    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), mask=tmp9, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp0 >= tmp7
     tmp12 = tl.full([1], 128, tl.int64)
     tmp13 = tmp0 < tmp12
     tmp14 = tmp11 & tmp13
-    tmp15 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), tmp14, eviction_policy='evict_last', other=0.0)
+    tmp15 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), mask=tmp14, eviction_policy='evict_last', other=0.0)
     tmp16 = tmp0 >= tmp12
     tmp17 = tl.full([1], 160, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr3 + (32*x1 + ((-128) + x0)), tmp19, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr3 + (32*x1 + ((-128) + x0)), mask=tmp19, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 192, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr4 + (32*x1 + ((-160) + x0)), tmp24, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr4 + (32*x1 + ((-160) + x0)), mask=tmp24, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 224, tl.int64)
     tmp28 = tmp0 < tmp27
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr5 + (32*x1 + ((-192) + x0)), tmp29, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr5 + (32*x1 + ((-192) + x0)), mask=tmp29, eviction_policy='evict_last', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1], 256, tl.int64)
     tmp33 = tmp0 < tmp32
-    tmp34 = tl.load(in_ptr6 + (32*x1 + ((-224) + x0)), tmp31, eviction_policy='evict_last', other=0.0)
+    tmp34 = tl.load(in_ptr6 + (32*x1 + ((-224) + x0)), mask=tmp31, eviction_policy='evict_last', other=0.0)
     tmp35 = tl.where(tmp29, tmp30, tmp34)
     tmp36 = tl.where(tmp24, tmp25, tmp35)
     tmp37 = tl.where(tmp19, tmp20, tmp36)
@@ -62,19 +63,12 @@

     tmp42 = tmp40 - tmp41
     tmp44 = tl.full([1], 1e-05, tl.float32)
     tmp45 = tmp43 + tmp44
-    tmp46 = tl.sqrt_rn(tmp45)
-    tmp47 = tl.full([1], 1, tl.int32)
-    tmp48 = (tmp47 / tmp46)
-    tmp49 = tl.full([1], 1.0, tl.float32)
-    tmp50 = tmp48 * tmp49
+    tmp50 = libdevice.rsqrt(tmp45)
     tmp51 = tmp42 * tmp50
-    tmp53 = tmp51 * tmp52
-    tmp55 = tmp53 + tmp54
+    tmp55 = libdevice.fma(tmp51, tmp52, tmp54)
     tmp56 = tl.full([1], 0, tl.int32)
     tmp57 = tl.maximum(tmp56, tmp55)
-    tl.store(in_out_ptr0 + (x2), tmp57, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp57, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
