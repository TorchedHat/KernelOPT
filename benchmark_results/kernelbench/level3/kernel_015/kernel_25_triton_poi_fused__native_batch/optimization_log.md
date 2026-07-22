# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24990 ns — **Best latency**: 15040 ns (**1.6616×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restrict_autotune_to_safe_num_warps_and_replace_fdiv_with_fmul_reciprocal`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_25`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_25_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (19600,1,1), Waves/SM: 9.28
- **Registers/thread**: 32
- **Duration**: 24990 ns
- **SM Throughput**: 74.35 % | **Memory**: 26.52 % | **DRAM**: 18.51 %
- **Achieved Occupancy**: 86.85 %
- **IPC Active**: 3.22 inst/cycle | **No Eligible**: 18.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.920× | 0.184× | ✓ |  |
| B | increase_xblock_and_autotune_for_fma_and_occupancy | 1.160× | 1.167× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_autotune | 1.003× | 0.184× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_compute_efficiency | 0.997× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24990 ns | 135780 ns | +110790 ns (+443.3%) | 21410 ns | -3580 ns (-14.3%) | 135710 ns | +110720 ns (+443.1%) | 25540 ns | +550 ns (+2.2%) |
| SM Throughput (%) | 74.35 % | 81.02 % | +6.67 pp | 71.83 % | -2.52 pp | 80.97 % | +6.62 pp | 74.50 % | +0.15 pp |
| Memory Throughput (%) | 26.52 % | 19.15 % | -7.37 pp | 31.83 % | +5.31 pp | 19.12 % | -7.40 pp | 26.56 % | +0.04 pp |
| DRAM Throughput (%) | 18.51 % | 3.39 % | -15.12 pp | 21.58 % | +3.07 pp | 3.40 % | -15.11 pp | 18.08 % | -0.43 pp |
| Achieved Occupancy (%) | 86.85 % | 68.09 % | -18.76 pp | 84.74 % | -2.11 pp | 68.10 % | -18.75 pp | 86.60 % | -0.25 pp |
| Registers/Thread | 32.00 register/thread | 38.00 register/thread | +6 | 32.00 register/thread | +0 | 38.00 register/thread | +6 | 32.00 register/thread | +0 |
| Waves/SM | 9.28 | 49.49 | +40.21 | 9.28 | +0.00 | 49.49 | +40.21 | 9.28 | +0.00 |
| No Eligible (%) | 18.88 % | 17.47 % | -1.41 pp | 19.88 % | +1.00 pp | 17.50 % | -1.38 pp | 18.22 % | -0.66 pp |
| IPC Active | 3.22 inst/cycle | 3.30 inst/cycle | +0.08 | 3.20 inst/cycle | -0.02 | 3.30 inst/cycle | +0.08 | 3.25 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.1672× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_avgpool_division_with_multiply_and_fuse_bn_epilogue_fma | 1.146× | 1.173× | ✓ |  |
| B | replace_sqrt_int_div_with_rsqrt_and_fma_in_bn_normalization | 1.154× | 1.167× | ✓ |  |
| C | replace_int_div_sqrt_with_rsqrt_and_fma_in_bn_normalization | 1.158× | 1.176× | ✓ |  |
| D | replace_int_div_sqrt_with_rsqrt_and_fma_in_bn_normalization | 1.160× | 1.173× | ✓ |  |
| E | add_triton_autotune | 1.149× | 1.194× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_avgpool_d... | Δ | replace_sqrt_int_... | Δ | replace_int_div_s... | Δ | replace_int_div_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24990 ns | 21310 ns | -3680 ns (-14.7%) | 21410 ns | -3580 ns (-14.3%) | 21250 ns | -3740 ns (-15.0%) | 21310 ns | -3680 ns (-14.7%) | 20930 ns | -4060 ns (-16.2%) |
| SM Throughput (%) | 74.35 % | 71.08 % | -3.27 pp | 71.68 % | -2.67 pp | 71.61 % | -2.74 pp | 71.29 % | -3.06 pp | 71.66 % | -2.69 pp |
| Memory Throughput (%) | 26.52 % | 31.49 % | +4.97 pp | 31.76 % | +5.24 pp | 31.71 % | +5.19 pp | 31.59 % | +5.07 pp | 31.75 % | +5.23 pp |
| DRAM Throughput (%) | 18.51 % | 21.72 % | +3.21 pp | 21.63 % | +3.12 pp | 21.86 % | +3.35 pp | 21.71 % | +3.20 pp | 21.90 % | +3.39 pp |
| Achieved Occupancy (%) | 86.85 % | 84.65 % | -2.20 pp | 84.33 % | -2.52 pp | 84.24 % | -2.61 pp | 84.71 % | -2.14 pp | 84.27 % | -2.58 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 9.28 | 9.28 | +0.00 | 9.28 | +0.00 | 9.28 | +0.00 | 9.28 | +0.00 | 9.28 | +0.00 |
| No Eligible (%) | 18.88 % | 19.82 % | +0.94 pp | 19.44 % | +0.56 pp | 20.47 % | +1.59 pp | 19.74 % | +0.86 pp | 19.50 % | +0.62 pp |
| IPC Active | 3.22 inst/cycle | 3.20 inst/cycle | -0.02 | 3.19 inst/cycle | -0.03 | 3.20 inst/cycle | -0.02 | 3.19 inst/cycle | -0.03 | 3.20 inst/cycle | -0.02 |

---

## Iteration 3

**Best so far**: 1.1672× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restrict_autotune_to_safe_num_warps_and_replace_fdiv_with_fmul_reciprocal | 1.269× | 1.662× | ✓ | **✓ best** |
| B | reduce_num_warps_to_cut_scheduler_contention_cpi_stall | 1.151× | 1.173× | ✓ |  |
| C | replace_avgpool_constant_division_with_reciprocal_multiply_for_fma_fusion | 1.160× | 1.174× | ✓ |  |
| D | replace_avgpool_division_with_reciprocal_multiply_and_remove_autotune | 1.150× | 1.169× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restrict_autotune... | Δ | reduce_num_warps_... | Δ | replace_avgpool_c... | Δ | replace_avgpool_d... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24990 ns | 15040 ns | -9950 ns (-39.8%) | 21310 ns | -3680 ns (-14.7%) | 21280 ns | -3710 ns (-14.8%) | 21380 ns | -3610 ns (-14.4%) |
| SM Throughput (%) | 74.35 % | 34.71 % | -39.64 pp | 71.12 % | -3.23 pp | 71.48 % | -2.87 pp | 71.57 % | -2.78 pp |
| Memory Throughput (%) | 26.52 % | 30.55 % | +4.03 pp | 31.49 % | +4.97 pp | 31.66 % | +5.14 pp | 31.70 % | +5.18 pp |
| DRAM Throughput (%) | 18.51 % | 30.55 % | +12.04 pp | 21.56 % | +3.05 pp | 21.58 % | +3.07 pp | 21.44 % | +2.93 pp |
| Achieved Occupancy (%) | 86.85 % | 20.16 % | -66.69 pp | 85.21 % | -1.64 pp | 85.07 % | -1.78 pp | 84.80 % | -2.05 pp |
| Registers/Thread | 32.00 register/thread | 64.00 register/thread | +32 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 9.28 | 4.64 | -4.64 | 9.28 | +0.00 | 9.28 | +0.00 | 9.28 | +0.00 |
| No Eligible (%) | 18.88 % | 53.38 % | +34.50 pp | 20.08 % | +1.20 pp | 19.70 % | +0.82 pp | 19.40 % | +0.52 pp |
| IPC Active | 3.22 inst/cycle | 1.77 inst/cycle | -1.45 | 3.19 inst/cycle | -0.03 | 3.18 inst/cycle | -0.04 | 3.20 inst/cycle | -0.02 |


### Key Learnings

#### ✅ Plan A: `restrict_autotune_to_safe_num_warps_and_replace_fdiv_with_fmul_reciprocal` (1.662×)

Combining two changes — (1) restricting autotune configs to XBLOCK=128 with num_warps in [1,2,4] and (2) replacing division by constant 4.0 with multiplication by 0.25 — produced a 1.66x speedup (20.93 µs → 15.04 µs). The autotuner selected num_warps=1 (block_size=32), which dramatically reduced executed instructions from 11.8M to 4.1M and warp cycles per issued instruction from 17.06 to 7.39, indicating the smaller warp count caused the compiler to unroll or restructure the kernel more aggressively. The FDIV→FMUL change eliminates the MUFU.RCP+FMUL non-fuseable pair, contributing to the instruction count reduction. Note: achieved occupancy dropped from 84% to 20% and registers/thread doubled from 32 to 64, but the net effect was still a large speedup because the instruction count reduction dominated — this is a case where lower occupancy with fewer instructions wins over higher occupancy with more instructions.

> **Applicable when**: Kernel is compute-bound (Compute SOL > 60%), NCU FPInstructions rule fires with non-fused:fused FP32 ratio >= 3:1, autotune configs include XBLOCK values that risk register spill, and the kernel contains division by a power-of-2 constant. Varying num_warps at fixed XBLOCK is beneficial when the scheduler shows high 'No Eligible' percentage (>15%) suggesting warp scheduling inefficiency.

> **Avoid when**: Do not restrict num_warps to very low values (e.g., num_warps=1) when the kernel is memory-latency bound and needs many warps to hide memory latency — the after profile shows CPIStall at 46.3% from L1TEX stalls, indicating this kernel is now latency-limited. Also do not apply when the kernel already has low register pressure and can benefit from higher occupancy to hide latency.

---

## Iteration 4

**Best so far**: 1.6616× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_at_fixed_xblock | 1.165× | 1.178× | ✓ |  |
| B | change_large_streaming_loads_to_evict_first_to_protect_bn_params_in_l2 | 1.164× | 1.174× | ✓ |  |
| C | pipeline_loads_with_num_stages_to_reduce_cpi_stall_contention | 1.145× | 1.171× | ✓ |  |
| D | add_evict_first_to_store_for_l2_compression | 1.126× | 1.130× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | change_large_stre... | Δ | pipeline_loads_wi... | Δ | add_evict_first_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24990 ns | 21220 ns | -3770 ns (-15.1%) | 21280 ns | -3710 ns (-14.8%) | 21340 ns | -3650 ns (-14.6%) | 22110 ns | -2880 ns (-11.5%) |
| SM Throughput (%) | 74.35 % | 71.44 % | -2.91 pp | 71.31 % | -3.04 pp | 71.88 % | -2.47 pp | 71.28 % | -3.07 pp |
| Memory Throughput (%) | 26.52 % | 31.70 % | +5.18 pp | 31.55 % | +5.03 pp | 31.82 % | +5.30 pp | 30.43 % | +3.91 pp |
| DRAM Throughput (%) | 18.51 % | 21.62 % | +3.11 pp | 21.53 % | +3.02 pp | 21.49 % | +2.98 pp | 20.76 % | +2.25 pp |
| Achieved Occupancy (%) | 86.85 % | 84.89 % | -1.96 pp | 84.78 % | -2.07 pp | 84.67 % | -2.18 pp | 85.20 % | -1.65 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 9.28 | 9.28 | +0.00 | 9.28 | +0.00 | 9.28 | +0.00 | 9.28 | +0.00 |
| No Eligible (%) | 18.88 % | 19.44 % | +0.56 pp | 19.79 % | +0.91 pp | 19.79 % | +0.91 pp | 20.38 % | +1.50 pp |
| IPC Active | 3.22 inst/cycle | 3.18 inst/cycle | -0.04 | 3.20 inst/cycle | -0.02 | 3.20 inst/cycle | -0.02 | 3.17 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `increase_num_warps_to_hide_l1tex_latency_at_fixed_xblock` (1.178×)

When a Triton kernel runs with very few warps per block (e.g., num_warps=1, block_size=32), the scheduler has no other warps to issue while one warp stalls on L1TEX memory latency, leaving 53% of scheduler cycles with no eligible warp (CPIStall rule: 46.3% of stall cycles from L1TEX). Removing low num_warps configs (1, 2) from the autotune set and adding higher ones (4, 8, 16) forces the autotuner to select a latency-hiding configuration: the autotuner selected block_size=128 (num_warps=4), raising eligible warps/scheduler from 0.85 to 8.05, reducing 'No Eligible' from 53% to 19%, and lifting achieved occupancy from 20% to 85%. Although elapsed cycles increased (22,392 → 31,634) due to 3x more executed instructions, the wall-clock speedup of 1.18x reflects that the prior low-warp config was a local minimum — fewer instructions but severe latency exposure — while the higher-warp config hides latency at the cost of more instructions but achieves better overall throughput.

> **Applicable when**: NCU CPIStall rule fires with L1TEX stalls accounting for >30% of warp cycles; Scheduler Statistics show 'No Eligible' > 40% and Eligible Warps Per Scheduler < 2.0; Achieved Occupancy is well below Theoretical Occupancy (e.g., 20% vs 50%); the low occupancy is caused by too few warps per block (block_size=32, num_warps=1) rather than register pressure or shared memory limits; the autotune config set only includes low num_warps values.

> **Avoid when**: The kernel already achieves high occupancy (>70%) and high eligible warps per scheduler (>5); the low warp count was intentional to reduce register pressure and the resulting instruction count reduction dominates (see exp_003 where num_warps=1 gave 1.66x speedup by reducing instructions 65%); the kernel is purely compute-bound with no memory latency stalls; increasing num_warps would cause register spill (registers/thread would exceed SM limits). Note: this strategy is in tension with exp_003 — the optimal num_warps depends on whether instruction-count reduction or latency-hiding dominates for the specific kernel state.

---

## Iteration 5

**Best so far**: 1.6616× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_low_num_warps_and_larger_xblock_for_instruction_reduction | 1.258× | 1.159× | ✓ |  |
| B | cap_registers_with_maxnreg_to_reduce_not_selected_warp_scheduler_contention | 1.121× | 1.132× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 24990 ns | 21570 ns | -3420 ns (-13.7%) | 22080 ns | -2910 ns (-11.6%) |
| SM Throughput (%) | 74.35 % | 71.17 % | -3.18 pp | 70.97 % | -3.38 pp |
| Memory Throughput (%) | 26.52 % | 31.52 % | +5.00 pp | 30.29 % | +3.77 pp |
| DRAM Throughput (%) | 18.51 % | 21.29 % | +2.78 pp | 20.78 % | +2.27 pp |
| Achieved Occupancy (%) | 86.85 % | 84.82 % | -2.03 pp | 85.28 % | -1.57 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 9.28 | 9.28 | +0.00 | 9.28 | +0.00 |
| No Eligible (%) | 18.88 % | 19.49 % | +0.61 pp | 20.43 % | +1.55 pp |
| IPC Active | 3.22 inst/cycle | 3.21 inst/cycle | -0.01 | 3.14 inst/cycle | -0.08 |

---

## Cumulative Progress

```
Baseline:  24990 ns  (1.000×)
Iter 1     21410 ns  (1.167×)  — increase_xblock_and_autotune_for_fma_and_occupancy
Iter 3     15040 ns  (1.662×)  — restrict_autotune_to_safe_num_warps_and_replace_fdiv_with_fmul_reciprocal

Final best: 15040 ns  (1.6616×)
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
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=16),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2508800
@@ -16,55 +29,54 @@

     x2 = xindex // 8960
     x3 = xindex // 320
     x4 = xindex
-    tmp51 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
-    tmp53 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
-    tmp62 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
-    tmp64 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
+    tmp51 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp53 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp62 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp64 = tl.load(in_ptr10 + (x0), mask=xmask, eviction_policy='evict_last')
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
-    tmp12 = tl.full([1], 4.0, tl.float32)
-    tmp13 = (tmp11 / tmp12)
+    tmp13 = tmp11 * 0.25
     tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
     tmp15 = tl.where(tmp4, tmp13, tmp14)
     tmp16 = tmp0 >= tmp3
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
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), mask=tmp29 & xmask, eviction_policy='evict_last', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1], 256, tl.int64)
     tmp33 = tmp0 < tmp32
     tmp34 = tmp31 & tmp33
-    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-224) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-224) + x0)), mask=tmp34 & xmask, eviction_policy='evict_last', other=0.0)
     tmp36 = tmp0 >= tmp32
     tmp37 = tl.full([1], 288, tl.int64)
     tmp38 = tmp0 < tmp37
     tmp39 = tmp36 & tmp38
-    tmp40 = tl.load(in_ptr5 + (32*x3 + ((-256) + x0)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp40 = tl.load(in_ptr5 + (32*x3 + ((-256) + x0)), mask=tmp39 & xmask, eviction_policy='evict_last', other=0.0)
     tmp41 = tmp0 >= tmp37
     tmp42 = tl.full([1], 320, tl.int64)
     tmp43 = tmp0 < tmp42
-    tmp44 = tl.load(in_ptr6 + (32*x3 + ((-288) + x0)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp44 = tl.load(in_ptr6 + (32*x3 + ((-288) + x0)), mask=tmp41 & xmask, eviction_policy='evict_last', other=0.0)
     tmp45 = tl.where(tmp39, tmp40, tmp44)
     tmp46 = tl.where(tmp34, tmp35, tmp45)
     tmp47 = tl.where(tmp29, tmp30, tmp46)
@@ -74,19 +86,12 @@

     tmp52 = tmp50 - tmp51
     tmp54 = tl.full([1], 1e-05, tl.float32)
     tmp55 = tmp53 + tmp54
-    tmp56 = tl.sqrt_rn(tmp55)
-    tmp57 = tl.full([1], 1, tl.int32)
-    tmp58 = (tmp57 / tmp56)
-    tmp59 = tl.full([1], 1.0, tl.float32)
-    tmp60 = tmp58 * tmp59
-    tmp61 = tmp52 * tmp60
-    tmp63 = tmp61 * tmp62
-    tmp65 = tmp63 + tmp64
+    tmp56 = libdevice.rsqrt(tmp55)
+    tmp61 = tmp52 * tmp56
+    tmp65 = tl.fma(tmp61, tmp62, tmp64)
     tmp66 = tl.full([1], 0, tl.int32)
     tmp67 = tl.maximum(tmp66, tmp65)
-    tl.store(in_out_ptr0 + (x4), tmp67, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp67, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -103,7 +108,7 @@

     arg_11 = inputs[11]
     xnumel = 2508800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_25[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, 2508800, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_25[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, 2508800)
     return arg_11
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
