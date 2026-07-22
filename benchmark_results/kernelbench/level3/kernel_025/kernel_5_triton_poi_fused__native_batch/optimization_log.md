# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 123900 ns — **Best latency**: 135840 ns (**0.9121×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 29.70
- **Registers/thread**: 62
- **Duration**: 123900 ns
- **SM Throughput**: 61.67 % | **Memory**: 76.26 % | **DRAM**: 61.55 %
- **Achieved Occupancy**: 47.56 %
- **IPC Active**: 2.45 inst/cycle | **No Eligible**: 38.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_reduce_register_pressure | 1.026× | 0.891× | ✓ |  |
| B | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 4.153× | 0.996× | ✓ | **✓ best** |
| C | add_autotune_with_vectorized_tile_configs | 1.880× | 1.445× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | add_autotune_with... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 123900 ns | 139040 ns | +15140 ns (+12.2%) | 124420 ns | +520 ns (+0.4%) | 85730 ns | -38170 ns (-30.8%) |
| SM Throughput (%) | 61.67 % | 74.99 % | +13.32 pp | 59.37 % | -2.30 pp | 56.45 % | -5.22 pp |
| Memory Throughput (%) | 76.26 % | 24.59 % | -51.67 pp | 75.81 % | -0.45 pp | 79.70 % | +3.44 pp |
| DRAM Throughput (%) | 61.55 % | 12.16 % | -49.39 pp | 61.26 % | -0.29 pp | 10.61 % | -50.94 pp |
| Achieved Occupancy (%) | 47.56 % | 29.46 % | -18.10 pp | 47.55 % | -0.01 pp | 58.98 % | +11.42 pp |
| Registers/Thread | 62.00 register/thread | 96.00 register/thread | +34 | 62.00 register/thread | +0 | 42.00 register/thread | -20 |
| Waves/SM | 29.70 | 47.52 | +17.82 | 29.70 | +0.00 | 23.76 | -5.94 |
| No Eligible (%) | 38.69 % | 23.84 % | -14.85 pp | 37.81 % | -0.88 pp | 40.72 % | +2.03 pp |
| IPC Active | 2.45 inst/cycle | 3.04 inst/cycle | +0.59 | 2.43 inst/cycle | -0.02 | 2.37 inst/cycle | -0.08 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | change_eviction_policy_streaming_tensors_to_evict_first | 4.057× | 0.921× | ✓ |  |
| C | replace_autotune_configs_with_large_xblock_small_yblock_and_fused_rsqrt | 1.831× | 0.902× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_eviction_p... | Δ | replace_autotune_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 123900 ns | 134500 ns | +10600 ns (+8.6%) | 137280 ns | +13380 ns (+10.8%) |
| SM Throughput (%) | 61.67 % | 62.31 % | +0.64 pp | 65.34 % | +3.67 pp |
| Memory Throughput (%) | 76.26 % | 70.25 % | -6.01 pp | 73.45 % | -2.81 pp |
| DRAM Throughput (%) | 61.55 % | 56.61 % | -4.94 pp | 6.70 % | -54.85 pp |
| Achieved Occupancy (%) | 47.56 % | 47.48 % | -0.08 pp | 70.91 % | +23.35 pp |
| Registers/Thread | 62.00 register/thread | 62.00 register/thread | +0 | 40.00 register/thread | -22 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 39.60 | +9.90 |
| No Eligible (%) | 38.69 % | 37.93 % | -0.76 pp | 33.04 % | -5.65 pp |
| IPC Active | 2.45 inst/cycle | 2.51 inst/cycle | +0.06 | 2.68 inst/cycle | +0.23 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | replace_verbose_bn_arithmetic_with_rsqrt_and_fma | 4.809× | 0.912× | ✓ | **✓ best** |
| C | improve_coalescing_via_full_channel_tile_and_large_xblock | 3.243× | 3.267× | ✓ |  |
| E | add_triton_autotune | 1.442× | 0.928× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_verbose_b... | Δ | improve_coalescin... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 123900 ns | 135840 ns | +11940 ns (+9.6%) | 37920 ns | -85980 ns (-69.4%) | 133570 ns | +9670 ns (+7.8%) |
| SM Throughput (%) | 61.67 % | 31.08 % | -30.59 pp | 73.64 % | +11.97 pp | 61.79 % | +0.12 pp |
| Memory Throughput (%) | 76.26 % | 69.75 % | -6.51 pp | 39.49 % | -36.77 pp | 70.74 % | -5.52 pp |
| DRAM Throughput (%) | 61.55 % | 55.93 % | -5.62 pp | 8.84 % | -52.71 pp | 57.10 % | -4.45 pp |
| Achieved Occupancy (%) | 47.56 % | 59.99 % | +12.43 pp | 67.85 % | +20.29 pp | 47.54 % | -0.02 pp |
| Registers/Thread | 62.00 register/thread | 48.00 register/thread | -14 | 37.00 register/thread | -25 | 62.00 register/thread | +0 |
| Waves/SM | 29.70 | 23.76 | -5.94 | 19.80 | -9.90 | 29.70 | +0.00 |
| No Eligible (%) | 38.69 % | 68.53 % | +29.84 pp | 22.09 % | -16.60 pp | 36.54 % | -2.15 pp |
| IPC Active | 2.45 inst/cycle | 1.25 inst/cycle | -1.20 | 3.10 inst/cycle | +0.65 | 2.52 inst/cycle | +0.07 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_arithmetic_into_fma_chains_to_reduce_instruction_count | 3.222× | 3.279× | ✓ |  |
| C | expand_yblock_to_full_channel_tile_for_coalesced_nhwc_access | 3.196× | 3.290× | ✓ |  |
| D | add_triton_autotune | 3.241× | 3.218× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_ar... | Δ | expand_yblock_to_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 123900 ns | 37790 ns | -86110 ns (-69.5%) | 37660 ns | -86240 ns (-69.6%) | 38500 ns | -85400 ns (-68.9%) |
| SM Throughput (%) | 61.67 % | 73.55 % | +11.88 pp | 74.33 % | +12.66 pp | 72.96 % | +11.29 pp |
| Memory Throughput (%) | 76.26 % | 39.52 % | -36.74 pp | 39.75 % | -36.51 pp | 39.26 % | -37.00 pp |
| DRAM Throughput (%) | 61.55 % | 8.86 % | -52.69 pp | 8.87 % | -52.68 pp | 8.77 % | -52.78 pp |
| Achieved Occupancy (%) | 47.56 % | 67.90 % | +20.34 pp | 67.55 % | +19.99 pp | 67.91 % | +20.35 pp |
| Registers/Thread | 62.00 register/thread | 37.00 register/thread | -25 | 40.00 register/thread | -22 | 37.00 register/thread | -25 |
| Waves/SM | 29.70 | 19.80 | -9.90 | 19.80 | -9.90 | 19.80 | -9.90 |
| No Eligible (%) | 38.69 % | 23.19 % | -15.50 pp | 20.72 % | -17.97 pp | 22.81 % | -15.88 pp |
| IPC Active | 2.45 inst/cycle | 3.09 inst/cycle | +0.64 | 3.16 inst/cycle | +0.71 | 3.09 inst/cycle | +0.64 |


### Key Learnings

#### ✅ Plan A: `fuse_batchnorm_arithmetic_into_fma_chains_to_reduce_instruction_count` (3.279×)

Replacing separate MUL+ADD pairs in batch normalization scale/bias arithmetic with explicit tl.math.fma() calls produced no measurable improvement: non-fused FP32 instruction count remained identical (3,512,320 before and after), registers/thread stayed at 37, theoretical occupancy remained 75%, and duration changed by only 0.13 µs (37.92 → 37.79 µs). This indicates Triton's LLVM backend already auto-fuses eligible mul+add pairs into FMA instructions during compilation, making explicit tl.math.fma() calls redundant. The NCU FPInstructions rule may report non-fused instructions that are structurally unfusable (e.g., intermediate values with multiple consumers) rather than missed fusion opportunities.

> **Applicable when**: Never: this strategy does not produce measurable improvement when Triton's LLVM backend already handles FMA fusion automatically.

> **Avoid when**: Always avoid: explicit tl.math.fma() substitution for BN scale/bias arithmetic in Triton kernels does not reduce non-fused FP32 instruction count, register pressure, or kernel duration. The NCU FPInstructions rule reporting non-fused instructions does not imply that explicit fma() calls will help — the compiler already fuses what it can.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_with_large_flat_xblock_and_deep_pipeline_stages | 3.207× | 3.235× | ✓ |  |
| D | add_triton_autotune | 3.208× | 3.307× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 123900 ns | 38300 ns | -85600 ns (-69.1%) | 37470 ns | -86430 ns (-69.8%) |
| SM Throughput (%) | 61.67 % | 74.38 % | +12.71 pp | 74.84 % | +13.17 pp |
| Memory Throughput (%) | 76.26 % | 39.56 % | -36.70 pp | 39.89 % | -36.37 pp |
| DRAM Throughput (%) | 61.55 % | 8.74 % | -52.81 pp | 8.95 % | -52.60 pp |
| Achieved Occupancy (%) | 47.56 % | 67.58 % | +20.02 pp | 67.55 % | +19.99 pp |
| Registers/Thread | 62.00 register/thread | 40.00 register/thread | -22 | 40.00 register/thread | -22 |
| Waves/SM | 29.70 | 19.80 | -9.90 | 19.80 | -9.90 |
| No Eligible (%) | 38.69 % | 21.52 % | -17.17 pp | 21.92 % | -16.77 pp |
| IPC Active | 2.45 inst/cycle | 3.16 inst/cycle | +0.71 | 3.17 inst/cycle | +0.72 |


### Key Learnings

#### ❌ Plan A: `expand_autotune_with_large_flat_xblock_and_deep_pipeline_stages` (3.235×)

Expanding the autotune config set to include large flat tiles (XBLOCK=512/1024, YBLOCK=1) and deeper pipeline stages (num_stages=4/5) produced no improvement: all profiling metrics are essentially unchanged (Duration 37.66→38.3 µs, registers/thread still 40, theoretical occupancy still 75%, Block_Limit_Registers still 12). The YBLOCK=1 configs did not reduce register pressure as hypothesized — the compiler's register allocator did not eliminate YBLOCK-indexed live values when YBLOCK=1 was selected by autotune. When a kernel is already compute-bound at 74% SM throughput with register-limited occupancy, adding flat-tile or deeper-pipeline configs to the autotune search space does not help if the register count is determined by the kernel body arithmetic rather than the tile shape.

> **Applicable when**: Never: this strategy does not improve performance when the kernel is already compute-bound and register pressure is determined by kernel body arithmetic rather than tile shape.

> **Avoid when**: When a Triton kernel is compute-bound (SM throughput >70%) and register-limited (Block_Limit_Registers < Block_Limit_Warps), adding YBLOCK=1 flat-tile configs or num_stages=4/5 to the autotune search space does not reduce registers/thread or improve occupancy. The register count is set by the kernel body's live value count, not by the YBLOCK tile parameter. Avoid this approach when registers/thread is already 40+ and the kernel body contains many independent arithmetic chains (BN normalization, residual add, ReLU) that keep many values live simultaneously.

---

## Cumulative Progress

```
Baseline:  123900 ns  (1.000×)

Final best: 135840 ns  (0.9121×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 640
@@ -19,41 +31,32 @@

     y0 = (yindex % 64)
     y1 = yindex // 64
     y3 = yindex
-    tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), mask=xmask & ymask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), mask=xmask & ymask, eviction_policy='evict_first')
+    tmp17 = tl.load(in_ptr5 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr6 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr7 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr8 + (y0), mask=ymask, eviction_policy='evict_last')
+    # Branch 1: BN normalization with rsqrt + FMA
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1, 1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1, 1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    eps = tl.full([1, 1], 1e-05, tl.float32)
+    tmp_inv_std1 = libdevice.rsqrt(tmp3 + eps)
+    tmp_scale1 = tmp_inv_std1 * tmp12
+    tmp15 = tl.math.fma(tmp2, tmp_scale1, tmp14)
+    # Branch 2: BN normalization with rsqrt + FMA
     tmp18 = tmp16 - tmp17
-    tmp20 = tmp19 + tmp4
-    tmp21 = tl.sqrt_rn(tmp20)
-    tmp22 = (tmp7 / tmp21)
-    tmp23 = tmp22 * tmp9
-    tmp24 = tmp18 * tmp23
-    tmp26 = tmp24 * tmp25
-    tmp28 = tmp26 + tmp27
+    tmp_inv_std2 = libdevice.rsqrt(tmp19 + eps)
+    tmp_scale2 = tmp_inv_std2 * tmp25
+    tmp28 = tl.math.fma(tmp18, tmp_scale2, tmp27)
+    # Residual add + ReLU (unchanged)
     tmp29 = tmp15 + tmp28
     tmp30 = tl.full([1, 1], 0, tl.int32)
     tmp31 = tl.maximum(tmp30, tmp29)
-    tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, mask=xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -70,7 +73,7 @@

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
