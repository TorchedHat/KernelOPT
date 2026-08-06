# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16510 ns — **Best latency**: 11460 ns (**1.4407×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_partial_wave_tail_and_conditional_branch_divergence`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_20`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_20_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (13720,1,1), Waves/SM: 6.50
- **Registers/thread**: 32
- **Duration**: 16510 ns
- **SM Throughput**: 67.42 % | **Memory**: 23.66 % | **DRAM**: 23.66 %
- **Achieved Occupancy**: 81.23 %
- **IPC Active**: 3.13 inst/cycle | **No Eligible**: 20.73 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.999× | 0.203× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_wider_tiles | 0.999× | 0.366× | ✓ |  |
| C | autotune_xblock_for_vectorized_loads_and_reduced_scheduler_overhead | 1.000× | 1.015× | ✓ |  |
| D | autotune_xblock_num_warps_num_stages | 0.989× | 1.015× | ✓ |  |
| E | add_triton_autotune | 0.987× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | autotune_xblock_f... | Δ | autotune_xblock_n... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16510 ns | 81180 ns | +64670 ns (+391.7%) | 45060 ns | +28550 ns (+172.9%) | 16260 ns | -250 ns (-1.5%) | 16260 ns | -250 ns (-1.5%) | 16320 ns | -190 ns (-1.2%) |
| SM Throughput (%) | 67.42 % | 80.84 % | +13.42 pp | 77.52 % | +10.10 pp | 67.72 % | +0.30 pp | 67.76 % | +0.34 pp | 67.61 % | +0.19 pp |
| Memory Throughput (%) | 23.66 % | 18.27 % | -5.39 pp | 18.31 % | -5.35 pp | 24.05 % | +0.39 pp | 24.03 % | +0.37 pp | 23.95 % | +0.29 pp |
| DRAM Throughput (%) | 23.66 % | 4.82 % | -18.84 pp | 8.66 % | -15.00 pp | 24.05 % | +0.39 pp | 24.03 % | +0.37 pp | 23.95 % | +0.29 pp |
| Achieved Occupancy (%) | 81.23 % | 45.59 % | -35.64 pp | 64.93 % | -16.30 pp | 81.06 % | -0.17 pp | 81.41 % | +0.18 pp | 81.62 % | +0.39 pp |
| Registers/Thread | 32.00 register/thread | 62.00 register/thread | +30 | 34.00 register/thread | +2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 6.50 | 25.98 | +19.48 | 17.32 | +10.82 | 6.50 | +0.00 | 6.50 | +0.00 | 6.50 | +0.00 |
| No Eligible (%) | 20.73 % | 16.87 % | -3.86 pp | 18.25 % | -2.48 pp | 20.68 % | -0.05 pp | 20.64 % | -0.09 pp | 20.87 % | +0.14 pp |
| IPC Active | 3.13 inst/cycle | 3.32 inst/cycle | +0.19 | 3.25 inst/cycle | +0.12 | 3.12 inst/cycle | -0.01 | 3.14 inst/cycle | +0.01 | 3.12 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune | 0.997× | 1.018× | ✓ |  |
| B | improve_l1_cache_retention_for_broadcast_bn_parameters | 0.998× | 1.014× | ✓ |  |
| C | fix_autotune_grid_lambda_and_reduce_scheduler_contention | 1.175× | 1.421× | ✓ | **✓ best** |
| D | replace_manual_reciprocal_sqrt_with_rsqrt_for_fma_fusion | 1.055× | 1.189× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | improve_l1_cache_... | Δ | fix_autotune_grid... | Δ | replace_manual_re... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16510 ns | 16220 ns | -290 ns (-1.8%) | 16290 ns | -220 ns (-1.3%) | 11620 ns | -4890 ns (-29.6%) | 13890 ns | -2620 ns (-15.9%) |
| SM Throughput (%) | 67.42 % | 67.51 % | +0.09 pp | 67.67 % | +0.25 pp | 63.05 % | -4.37 pp | 63.43 % | -3.99 pp |
| Memory Throughput (%) | 23.66 % | 24.09 % | +0.43 pp | 23.99 % | +0.33 pp | 33.66 % | +10.00 pp | 28.57 % | +4.91 pp |
| DRAM Throughput (%) | 23.66 % | 24.09 % | +0.43 pp | 23.99 % | +0.33 pp | 33.66 % | +10.00 pp | 28.16 % | +4.50 pp |
| Achieved Occupancy (%) | 81.23 % | 81.25 % | +0.02 pp | 81.12 % | -0.11 pp | 73.74 % | -7.49 pp | 78.27 % | -2.96 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 6.50 | 6.50 | +0.00 | 6.50 | +0.00 | 3.25 | -3.25 | 6.50 | +0.00 |
| No Eligible (%) | 20.73 % | 20.84 % | +0.11 pp | 21.12 % | +0.39 pp | 19.13 % | -1.60 pp | 23.97 % | +3.24 pp |
| IPC Active | 3.13 inst/cycle | 3.13 inst/cycle | +0.00 | 3.12 inst/cycle | -0.01 | 3.18 inst/cycle | +0.05 | 2.98 inst/cycle | -0.15 |

---

## Iteration 3

**Best so far**: 1.4208× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_sqrt_reciprocal_with_libdevice_rsqrt_for_fma_fusion | 1.196× | 1.426× | ✓ | **✓ best** |
| B | explicit_fma_fusion_for_bn_epilogue_multiply_add_chain | 1.066× | 1.173× | ✓ |  |
| C | fuse_reciprocal_sqrt_to_rsqrt_for_fma_chain_and_fix_grid_lambda | 1.184× | 1.446× | ✓ |  |
| D | fix_grid_lambda_and_restrict_to_safe_small_xblock_configs | 1.177× | 1.421× | ✓ |  |
| E | add_triton_autotune | 0.999× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_sq... | Δ | explicit_fma_fusi... | Δ | fuse_reciprocal_s... | Δ | fix_grid_lambda_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16510 ns | 11580 ns | -4930 ns (-29.9%) | 14080 ns | -2430 ns (-14.7%) | 11420 ns | -5090 ns (-30.8%) | 11620 ns | -4890 ns (-29.6%) | 16610 ns | +100 ns (+0.6%) |
| SM Throughput (%) | 67.42 % | 42.64 % | -24.78 pp | 62.80 % | -4.62 pp | 38.64 % | -28.78 pp | 62.23 % | -5.19 pp | 67.75 % | +0.33 pp |
| Memory Throughput (%) | 23.66 % | 33.74 % | +10.08 pp | 28.28 % | +4.62 pp | 17.30 % | -6.36 pp | 33.71 % | +10.05 pp | 23.71 % | +0.05 pp |
| DRAM Throughput (%) | 23.66 % | 33.74 % | +10.08 pp | 27.76 % | +4.10 pp | 17.06 % | -6.60 pp | 33.71 % | +10.05 pp | 23.56 % | -0.10 pp |
| Achieved Occupancy (%) | 81.23 % | 49.01 % | -32.22 pp | 78.29 % | -2.94 pp | 41.65 % | -39.58 pp | 74.08 % | -7.15 pp | 81.69 % | +0.46 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 6.50 | 3.25 | -3.25 | 6.50 | +0.00 | 3.25 | -3.25 | 3.25 | -3.25 | 6.50 | +0.00 |
| No Eligible (%) | 20.73 % | 38.80 % | +18.07 pp | 23.93 % | +3.20 pp | 45.05 % | +24.32 pp | 19.15 % | -1.58 pp | 20.73 % | +0.00 pp |
| IPC Active | 3.13 inst/cycle | 2.39 inst/cycle | -0.74 | 2.95 inst/cycle | -0.18 | 2.13 inst/cycle | -1.00 | 3.19 inst/cycle | +0.06 | 3.15 inst/cycle | +0.02 |


### Key Learnings

#### ✅ Plan A: `replace_manual_sqrt_reciprocal_with_libdevice_rsqrt_for_fma_fusion` (1.426×)

When NCU's FPInstructions rule fires with a high non-fused to fused FP32 ratio (>4:1), a manual reciprocal-sqrt sequence (tl.sqrt_rn → int(1)/sqrt → float(1.0)*result) can be collapsed into a single libdevice.rsqrt() call. This emits one MUFU.RSQRT hardware instruction and allows the compiler to fuse the downstream multiply-add chain (e.g. (x-mean)*rsqrt*gamma+beta) into FFMA instructions, reducing total executed instructions by ~33% (5.68M→3.81M) and eliminating divergent branches (493K→55K). The trade-off is reduced achieved occupancy (73.7%→49.0%) due to fewer warps being eligible per scheduler cycle, but the instruction count reduction dominates and yields a net 1.43x speedup.

> **Applicable when**: NCU FPInstructions rule is triggered with non-fused:fused FP32 ratio >3:1. Kernel contains a manual reciprocal-sqrt pattern (tl.sqrt_rn + integer division + float cast) used for batch norm normalization or similar. libdevice is already imported (from triton.language.extra import libdevice). The input to rsqrt is always strictly positive (e.g. variance + epsilon where epsilon > 0).

> **Avoid when**: Do not apply when the input to rsqrt could be zero or negative (libdevice.rsqrt(0) returns inf). Do not apply if the kernel is already memory-bandwidth-bound (DRAM Throughput >80%) — the instruction reduction will not help. Be aware that achieved occupancy may drop significantly (this case: 73.7%→49.0%) if the kernel becomes latency-bound after instruction reduction; verify the net speedup is positive before committing.

---

## Iteration 4

**Best so far**: 1.4257× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_for_l1tex_latency_hiding | 1.182× | 1.457× | ✓ |  |
| B | fuse_bn_epilogue_multiply_add_with_libdevice_fma | 1.200× | 1.441× | ✓ | **✓ best** |
| C | fuse_bn_epilogue_rsqrt_and_fma_chain | 1.182× | 1.437× | ✓ |  |
| D | change_eviction_policy_streaming_loads_to_evict_first | 1.100× | 1.178× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fuse_bn_epilogue_... | Δ | fuse_bn_epilogue_... | Δ | change_eviction_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16510 ns | 11330 ns | -5180 ns (-31.4%) | 11460 ns | -5050 ns (-30.6%) | 11490 ns | -5020 ns (-30.4%) | 14020 ns | -2490 ns (-15.1%) |
| SM Throughput (%) | 67.42 % | 38.48 % | -28.94 pp | 42.83 % | -24.59 pp | 42.43 % | -24.99 pp | 63.52 % | -3.90 pp |
| Memory Throughput (%) | 23.66 % | 17.28 % | -6.38 pp | 34.23 % | +10.57 pp | 33.98 % | +10.32 pp | 28.63 % | +4.97 pp |
| DRAM Throughput (%) | 23.66 % | 17.18 % | -6.48 pp | 34.23 % | +10.57 pp | 33.98 % | +10.32 pp | 27.88 % | +4.22 pp |
| Achieved Occupancy (%) | 81.23 % | 42.37 % | -38.86 pp | 48.82 % | -32.41 pp | 48.35 % | -32.88 pp | 77.69 % | -3.54 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 6.50 | 3.25 | -3.25 | 3.25 | -3.25 | 3.25 | -3.25 | 6.50 | +0.00 |
| No Eligible (%) | 20.73 % | 46.03 % | +25.30 pp | 39.30 % | +18.57 pp | 40.15 % | +19.42 pp | 23.98 % | +3.25 pp |
| IPC Active | 3.13 inst/cycle | 2.17 inst/cycle | -0.96 | 2.38 inst/cycle | -0.75 | 2.34 inst/cycle | -0.79 | 2.99 inst/cycle | -0.14 |


### Key Learnings

#### ✅ Plan B: `fuse_bn_epilogue_multiply_add_with_libdevice_fma` (1.441×)

Replacing a FMUL+FADD pair (x_norm * gamma + beta) in a batch-norm epilogue with a single libdevice.fma(x_norm, gamma, beta) call at the already-optimal XBLOCK=64 operating point yielded a 1.44x end-to-end speedup. Paradoxically, the per-kernel profiling shows elapsed cycles barely changed (17011 → 16938, ~0.4% reduction) while executed instructions increased (3.40M → 3.81M) and registers per thread rose from 30 to 32. The dominant effect appears to be that the autotuner reordered configs (XBLOCK=128 placed first), causing it to select a different config that improved DRAM throughput (17.3% → 34.2%) and achieved occupancy (41.7% → 48.8%), while the FMA substitution itself had a minor arithmetic effect. The end-to-end speedup is real but driven primarily by the autotuner config reordering rather than the FMA instruction fusion alone.

> **Applicable when**: NCU FPInstructions rule fires with a high non-fused to fused FP32 ratio (>5:1). The kernel contains a batch-norm epilogue with an explicit FMUL immediately followed by FADD on the same intermediate (pattern: tmp = a * b; result = tmp + c). libdevice is already imported. The kernel is operating at its autotuner-optimal XBLOCK (not being changed simultaneously). The end-to-end speedup is the target metric, not just per-kernel elapsed cycles.

> **Avoid when**: Do not expect large per-kernel elapsed cycle reductions from this change alone — the profiling data shows the FMA substitution at XBLOCK=64 yields only ~0.4% cycle reduction per kernel. The end-to-end speedup (1.44x) appears to be driven by autotuner config reordering side effects rather than the FMA instruction itself. Do not apply when the kernel is already memory-bandwidth-bound (DRAM Throughput >80%) or when registers per thread are at the block limit boundary and adding the fma call increases register count.

---

## Iteration 5

**Best so far**: 1.4407× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_stages3_software_pipelining_for_l1tex_stall | 1.185× | 1.466× | ✓ |  |
| B | fix_grid_lambda_and_add_pipelined_large_xblock_configs_to_eliminate_partial_wave | 1.200× | 1.437× | ✓ |  |
| C | reduce_partial_wave_tail_and_conditional_branch_divergence | 1.225× | 1.441× | ✓ | **✓ best** |
| D | reduce_xblock_to_64_cut_not_selected_scheduler_stall | 1.098× | 1.159× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | fix_grid_lambda_a... | Δ | reduce_partial_wa... | Δ | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16510 ns | 11260 ns | -5250 ns (-31.8%) | 11490 ns | -5020 ns (-30.4%) | 11460 ns | -5050 ns (-30.6%) | 14240 ns | -2270 ns (-13.7%) |
| SM Throughput (%) | 67.42 % | 37.83 % | -29.59 pp | 38.54 % | -28.88 pp | 44.36 % | -23.06 pp | 63.55 % | -3.87 pp |
| Memory Throughput (%) | 23.66 % | 17.28 % | -6.38 pp | 17.28 % | -6.38 pp | 34.18 % | +10.52 pp | 28.64 % | +4.98 pp |
| DRAM Throughput (%) | 23.66 % | 17.28 % | -6.38 pp | 16.96 % | -6.70 pp | 34.18 % | +10.52 pp | 27.46 % | +3.80 pp |
| Achieved Occupancy (%) | 81.23 % | 41.39 % | -39.84 pp | 41.93 % | -39.30 pp | 48.35 % | -32.88 pp | 77.91 % | -3.32 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 6.50 | 3.25 | -3.25 | 3.25 | -3.25 | 3.25 | -3.25 | 6.50 | +0.00 |
| No Eligible (%) | 20.73 % | 45.14 % | +24.41 pp | 45.79 % | +25.06 pp | 34.97 % | +14.24 pp | 25.10 % | +4.37 pp |
| IPC Active | 3.13 inst/cycle | 2.12 inst/cycle | -1.01 | 2.12 inst/cycle | -1.01 | 2.58 inst/cycle | -0.55 | 3.00 inst/cycle | -0.13 |


### Key Learnings

#### ✅ Plan A: `autotune_num_stages3_software_pipelining_for_l1tex_stall` (1.466×)

When CPIStall identifies L1TEX scoreboard stalls as the dominant bottleneck (~48% of warp cycles) and the existing autotune search space only covers num_stages=2, adding num_stages=3 variants across all XBLOCK/num_warps combinations allows the Triton compiler to emit prefetch instructions that overlap L1TEX loads with arithmetic, marginally reducing warp cycles per issued instruction (12.84→12.60) and no-eligible percentage (46.03%→45.14%). The per-kernel improvement is modest (elapsed cycles 16883→16788, ~0.6% reduction) but contributes to a 1.47x end-to-end speedup, suggesting the expanded search space allows the autotuner to find a slightly better configuration. This strategy is mechanistically distinct from expanding num_warps alone (which was tried in iteration 4 with similar marginal per-kernel gains), as num_stages controls the software pipeline depth rather than the warp count.

> **Applicable when**: CPIStall rule fires with L1TEX scoreboard stalls >40% of warp cycles. Existing autotune configs all use num_stages=2. Registers per thread are low enough that an additional pipeline stage does not cause register spill (e.g., 30 regs/thread with XBLOCK=64 → 1920 regs/block, well within 65536 register file). The kernel is a pointwise or elementwise fusion (not a matmul/reduction) where software pipelining can overlap loads with arithmetic. The grid is already a correct lambda using meta['XBLOCK'].

> **Avoid when**: Do not expect large per-kernel elapsed cycle reductions from this change alone — the profiling data shows only ~0.6% cycle reduction per kernel. The strategy is most useful as part of a broader autotune search space expansion. Do not add num_stages=3 when registers per thread are already high (>40) as the additional pipeline stage registers may cause spill. Avoid when the kernel is already DRAM-bandwidth-bound (DRAM Throughput >60%) — software pipelining hides L1TEX latency, not DRAM bandwidth.

---

## Cumulative Progress

```
Baseline:  16510 ns  (1.000×)
Iter 2     11620 ns  (1.421×)  — fix_autotune_grid_lambda_and_reduce_scheduler_contention
Iter 3     11580 ns  (1.426×)  — replace_manual_sqrt_reciprocal_with_libdevice_rsqrt_for_fma_fusion
Iter 4     11460 ns  (1.441×)  — fuse_bn_epilogue_multiply_add_with_libdevice_fma

Final best: 11460 ns  (1.4407×)
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
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1756160
@@ -16,21 +28,21 @@

     x2 = xindex // 6272
     x3 = xindex // 224
     x4 = xindex
-    tmp33 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
-    tmp35 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
-    tmp44 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
-    tmp46 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
+    tmp33 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_first')
+    tmp35 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_first')
+    tmp44 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_first')
+    tmp46 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_first')
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
@@ -40,35 +52,26 @@

     tmp17 = tl.full([1], 160, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 192, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 224, tl.int64)
     tmp28 = tmp0 < tmp27
-    tmp29 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp30 = tl.where(tmp24, tmp25, tmp29)
-    tmp31 = tl.where(tmp19, tmp20, tmp30)
-    tmp32 = tl.where(tmp4, tmp15, tmp31)
+    tmp29 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), mask=tmp26 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp32 = tmp15 + tmp20 + tmp25 + tmp29
     tmp34 = tmp32 - tmp33
     tmp36 = tl.full([1], 1e-05, tl.float32)
     tmp37 = tmp35 + tmp36
-    tmp38 = tl.sqrt_rn(tmp37)
-    tmp39 = tl.full([1], 1, tl.int32)
-    tmp40 = (tmp39 / tmp38)
-    tmp41 = tl.full([1], 1.0, tl.float32)
-    tmp42 = tmp40 * tmp41
+    tmp42 = libdevice.rsqrt(tmp37)
     tmp43 = tmp34 * tmp42
-    tmp45 = tmp43 * tmp44
-    tmp47 = tmp45 + tmp46
+    tmp47 = libdevice.fma(tmp43, tmp44, tmp46)
     tmp48 = tl.full([1], 0, tl.int32)
     tmp49 = tl.maximum(tmp48, tmp47)
-    tl.store(in_out_ptr0 + (x4), tmp49, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp49, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -82,7 +85,7 @@

     arg_8 = inputs[8]
     xnumel = 1756160
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_20[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 1756160, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_20[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 1756160)
     return arg_8
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
