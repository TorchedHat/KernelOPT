# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6560 ns — **Best latency**: 5980 ns (**1.0970×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`restore_small_xblock_configs_and_fuse_rsqrt`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_45_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4410,1,1), Waves/SM: 2.09
- **Registers/thread**: 28
- **Duration**: 6560 ns
- **SM Throughput**: 48.91 % | **Memory**: 25.86 % | **DRAM**: 25.86 %
- **Achieved Occupancy**: 73.38 %
- **IPC Active**: 2.87 inst/cycle | **No Eligible**: 28.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave_tail | 0.957× | 0.940× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave | 0.998× | 1.009× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.840× | 0.732× | ✓ |  |
| D | add_autotune_eliminate_partial_wave | 0.852× | 0.743× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | add_autotune_elim... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 6980 ns | +420 ns (+6.4%) | 6500 ns | -60 ns (-0.9%) | 8960 ns | +2400 ns (+36.6%) | 8830 ns | +2270 ns (+34.6%) |
| SM Throughput (%) | 48.91 % | 55.89 % | +6.98 pp | 48.71 % | -0.20 pp | 61.84 % | +12.93 pp | 61.94 % | +13.03 pp |
| Memory Throughput (%) | 25.86 % | 24.23 % | -1.63 pp | 26.11 % | +0.25 pp | 18.87 % | -6.99 pp | 19.18 % | -6.68 pp |
| DRAM Throughput (%) | 25.86 % | 24.23 % | -1.63 pp | 26.11 % | +0.25 pp | 18.87 % | -6.99 pp | 19.18 % | -6.68 pp |
| Achieved Occupancy (%) | 73.38 % | 75.45 % | +2.07 pp | 73.41 % | +0.03 pp | 77.06 % | +3.68 pp | 76.82 % | +3.44 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 2.09 | 2.09 | +0.00 | 2.09 | +0.00 | 2.09 | +0.00 | 2.09 | +0.00 |
| No Eligible (%) | 28.33 % | 19.51 % | -8.82 pp | 26.09 % | -2.24 pp | 18.61 % | -9.72 pp | 19.52 % | -8.81 pp |
| IPC Active | 2.87 inst/cycle | 3.15 inst/cycle | +0.28 | 2.88 inst/cycle | +0.01 | 3.22 inst/cycle | +0.35 | 3.24 inst/cycle | +0.37 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_reciprocal_with_rsqrt_for_fma_fusion | 1.020× | 1.119× | ✓ |  |
| B | replace_sqrt_div_chain_with_rsqrt_to_enable_fma_fusion | 0.966× | 1.046× | ✓ |  |
| C | restore_small_xblock_configs_and_fuse_rsqrt | 1.042× | 1.097× | ✓ | **✓ best** |
| D | fuse_fp32_multiply_add_to_fma_in_batchnorm_epilogue | 1.009× | 0.940× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_reci... | Δ | replace_sqrt_div_... | Δ | restore_small_xbl... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 5860 ns | -700 ns (-10.7%) | 6270 ns | -290 ns (-4.4%) | 5980 ns | -580 ns (-8.8%) | 6980 ns | +420 ns (+6.4%) |
| SM Throughput (%) | 48.91 % | 42.16 % | -6.75 pp | 44.88 % | -4.03 pp | 22.98 % | -25.93 pp | 50.35 % | +1.44 pp |
| Memory Throughput (%) | 25.86 % | 28.91 % | +3.05 pp | 27.07 % | +1.21 pp | 28.35 % | +2.49 pp | 24.27 % | -1.59 pp |
| DRAM Throughput (%) | 25.86 % | 28.91 % | +3.05 pp | 27.07 % | +1.21 pp | 28.35 % | +2.49 pp | 24.27 % | -1.59 pp |
| Achieved Occupancy (%) | 73.38 % | 70.35 % | -3.03 pp | 69.84 % | -3.54 pp | 61.70 % | -11.68 pp | 56.92 % | -16.46 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 34.00 register/thread | +6 |
| Waves/SM | 2.09 | 2.09 | +0.00 | 2.09 | +0.00 | 1.04 | -1.05 | 2.78 | +0.69 |
| No Eligible (%) | 28.33 % | 31.56 % | +3.23 pp | 26.55 % | -1.78 pp | 55.02 % | +26.69 pp | 27.66 % | -0.67 pp |
| IPC Active | 2.87 inst/cycle | 2.64 inst/cycle | -0.23 | 2.73 inst/cycle | -0.14 | 1.75 inst/cycle | -1.12 | 2.78 inst/cycle | -0.09 |

---

## Iteration 3

**Best so far**: 1.097× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_for_streaming_loads_to_preserve_broadcast_vectors_in_l1 | 0.954× | 1.114× | ✓ |  |
| B | change_eviction_policy_streaming_loads_to_evict_first | 0.958× | 1.114× | ✓ |  |
| C | fuse_batchnorm_epilogue_with_fma_and_fix_eviction_policy | 0.964× | 1.056× | ✓ |  |
| D | fix_broken_autotune_grid_and_add_small_xblock_configs | 1.011× | 1.103× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_for_s... | Δ | change_eviction_p... | Δ | fuse_batchnorm_ep... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 5890 ns | -670 ns (-10.2%) | 5890 ns | -670 ns (-10.2%) | 6210 ns | -350 ns (-5.3%) | 5950 ns | -610 ns (-9.3%) |
| SM Throughput (%) | 48.91 % | 23.59 % | -25.32 pp | 23.23 % | -25.68 pp | 22.07 % | -26.84 pp | 22.10 % | -26.81 pp |
| Memory Throughput (%) | 25.86 % | 28.86 % | +3.00 pp | 28.80 % | +2.94 pp | 27.31 % | +1.45 pp | 28.57 % | +2.71 pp |
| DRAM Throughput (%) | 25.86 % | 28.86 % | +3.00 pp | 28.80 % | +2.94 pp | 27.31 % | +1.45 pp | 28.57 % | +2.71 pp |
| Achieved Occupancy (%) | 73.38 % | 61.50 % | -11.88 pp | 62.52 % | -10.86 pp | 63.03 % | -10.35 pp | 61.29 % | -12.09 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 2.09 | 1.04 | -1.05 | 1.04 | -1.05 | 1.04 | -1.05 | 1.04 | -1.05 |
| No Eligible (%) | 28.33 % | 53.82 % | +25.49 pp | 53.86 % | +25.53 pp | 53.64 % | +25.31 pp | 55.57 % | +27.24 pp |
| IPC Active | 2.87 inst/cycle | 1.77 inst/cycle | -1.10 | 1.77 inst/cycle | -1.10 | 1.77 inst/cycle | -1.10 | 1.77 inst/cycle | -1.10 |


### Key Learnings

#### ✅ Plan B: `change_eviction_policy_streaming_loads_to_evict_first` (1.114×)

Changing eviction_policy from 'evict_last' to 'evict_first' on streaming (use-once) global loads did not improve L1/TEX hit rate (remained at 43.05%) and the kernel duration was essentially unchanged (5.86 µs → 5.89 µs). The apparent 1.11x speedup is attributable to the concurrent autotune config change (block_size 128 → 64, halving executed instructions from 1.9M to 1.08M) rather than the eviction policy change itself. The eviction policy change alone did not reduce L1TEX stall cycles — eligible warps per scheduler actually dropped from 4.36 to 2.40 and warp cycles per issued instruction worsened from 16.65 to 25.76, indicating the smaller block size reduced latency-hiding parallelism. The speedup is therefore driven by the autotune config selection, not the evict_first hint.

> **Applicable when**: Kernel mixes streaming (use-once) loads with broadcast/reused loads; L1/TEX hit rate is moderate (30-60%); CPIStall rule fires with L1TEX stall as dominant stall type; the reused data working set fits in L1 (e.g., <64KB); autotune config space includes smaller block sizes that may be selected concurrently.

> **Avoid when**: The eviction policy change is the only change being made and the L1/TEX hit rate does not improve after the change (as seen here — hit rate stayed at 43.05%); when the kernel is already at low block size (64 threads) where reducing warps per SM hurts latency hiding more than eviction policy helps cache efficiency; when the dominant stall is not L1TEX-miss-driven but rather 'not selected' (scheduler contention).

---

## Iteration 4

**Best so far**: 1.097× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_stages_to_pipeline_memory_loads_and_hide_l1tex_latency | 0.936× | 1.079× | ✓ |  |
| B | add_num_stages_pipelining_to_autotune_configs_to_hide_l1tex_latency | 0.943× | 1.051× | ✓ |  |
| C | convert_avgpool_division_to_reciprocal_multiply_for_fma_chain_fusion | 1.040× | 1.051× | ✓ |  |
| D | replace_avg_pool_division_with_multiply_to_enable_fma_fusion | 0.967× | 1.079× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | add_num_stages_pi... | Δ | convert_avgpool_d... | Δ | replace_avg_pool_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 6080 ns | -480 ns (-7.3%) | 6240 ns | -320 ns (-4.9%) | 6240 ns | -320 ns (-4.9%) | 6080 ns | -480 ns (-7.3%) |
| SM Throughput (%) | 48.91 % | 23.85 % | -25.06 pp | 22.92 % | -25.99 pp | 23.67 % | -25.24 pp | 23.48 % | -25.43 pp |
| Memory Throughput (%) | 25.86 % | 27.83 % | +1.97 pp | 27.18 % | +1.32 pp | 27.20 % | +1.34 pp | 27.93 % | +2.07 pp |
| DRAM Throughput (%) | 25.86 % | 27.83 % | +1.97 pp | 27.18 % | +1.32 pp | 27.20 % | +1.34 pp | 27.93 % | +2.07 pp |
| Achieved Occupancy (%) | 73.38 % | 61.09 % | -12.29 pp | 61.43 % | -11.95 pp | 63.11 % | -10.27 pp | 63.21 % | -10.17 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 2.09 | 1.04 | -1.05 | 1.04 | -1.05 | 1.04 | -1.05 | 1.04 | -1.05 |
| No Eligible (%) | 28.33 % | 53.50 % | +25.17 pp | 59.98 % | +31.65 pp | 54.35 % | +26.02 pp | 53.33 % | +25.00 pp |
| IPC Active | 2.87 inst/cycle | 1.75 inst/cycle | -1.12 | 1.77 inst/cycle | -1.10 | 1.66 inst/cycle | -1.21 | 1.69 inst/cycle | -1.18 |


### Key Learnings

#### ✅ Plan A: `autotune_num_stages_to_pipeline_memory_loads_and_hide_l1tex_latency` (1.079×)

Adding num_stages=2 and num_stages=3 configs to the @triton.autotune sweep of a flat pointwise kernel (no inner reduction loop) did not reduce L1TEX stall cycles or improve eligible warps per scheduler — warp cycles per issued instruction actually worsened from 21.74 to 24.12 and eligible warps dropped from 2.56 to 2.42. The L1/TEX hit rate, L2 hit rate, and executed instruction count were all unchanged, confirming that num_stages pipelining has no meaningful effect on pointwise kernels where there is no inner loop to pipeline across. The 7.9% wall-clock speedup is inconsistent with the per-kernel profiling showing longer elapsed cycles (8714 → 9043), suggesting the speedup is attributable to system-level effects (e.g., reduced kernel launch overhead, better SM scheduling across the full model) rather than the num_stages change itself.

> **Applicable when**: Kernel has an explicit inner reduction or tiling loop where num_stages maps to loop-level software pipelining (e.g., GEMM, conv, reduction kernels). CPIStall rule fires with L1TEX stalls as dominant bottleneck AND the kernel has multiple loop iterations per block where async prefetch can overlap with compute.

> **Avoid when**: Kernel is a flat pointwise kernel with no inner loop — num_stages>1 has no effect on the outer program-ID loop in Triton for pointwise kernels, and the autotuner will simply select the same XBLOCK/num_warps config regardless of num_stages. Do not apply when per-kernel profiling shows elapsed cycles increasing after the change (as seen here: 8714 → 9043 cycles), even if wall-clock speedup appears positive due to system-level confounds. Do not apply when L1/TEX hit rate and executed instruction count are unchanged between before and after.

#### ✅ Plan B: `add_num_stages_pipelining_to_autotune_configs_to_hide_l1tex_latency` (1.051×)

Adding num_stages=2 and num_stages=3 variants to the @triton.autotune config list for a pointwise kernel did not reduce L1TEX stall cycles and actually worsened scheduler efficiency: No Eligible cycles increased from 53.82% to 59.98%, eligible warps per scheduler dropped from 2.56 to 2.11, and warp cycles per issued instruction worsened from 21.74 to 23.7. The L1/TEX hit rate remained unchanged at 43.05% and executed instructions were identical (1.08M), confirming that Triton's num_stages parameter has no meaningful software-pipelining effect on flat pointwise kernels without explicit reduction loops — the compiler does not generate prefetch code for simple element-wise access patterns. The 5.1% wall-clock speedup is inconsistent with the per-kernel profiling showing longer duration (5.89 µs → 6.24 µs), suggesting the speedup was measured at the end-to-end level and may reflect autotuner selection of a different config (e.g., a num_stages=1 config with a different XBLOCK) rather than any benefit from num_stages>1.

> **Applicable when**: Kernel has explicit tiled reduction loops (matmul, conv) where Triton can generate async prefetch code across loop iterations; num_stages>1 is meaningful when there is a loop body with independent load and compute phases that can be overlapped.

> **Avoid when**: Kernel is a flat pointwise kernel without explicit reduction loops (element-wise ops, batch-norm inference, pooling, concat fusions); in these cases num_stages>1 in autotune configs does not generate prefetch code and may cause the autotuner to select a suboptimal config. The per-kernel profiling showed duration increased (5.89 µs → 6.24 µs) despite the reported end-to-end speedup, indicating the num_stages>1 configs were not the source of improvement. Do not add num_stages>1 configs when CPIStall is driven by L1TEX stalls in a pointwise kernel — the stalls cannot be hidden by software pipelining without a loop structure.

#### ✅ Plan C: `convert_avgpool_division_to_reciprocal_multiply_for_fma_chain_fusion` (1.051×)

Replacing a tl.full([1], 4.0) division with a literal 0.25 multiply (to enable FMA fusion) and adding num_stages=2 autotune configs did not reduce executed instructions (1.076M before and after) and did not improve warp cycles per issued instruction (21.74 both before and after). The NCU FPInstructions rule predicted up to 44% FP32 throughput improvement, but the profiling shows no measurable change in instruction count or compute throughput (23.59% → 23.67%). The elapsed cycles actually increased slightly (8714 → 9255), and the 1.05x speedup is likely attributable to wall-clock measurement noise or the autotune config expansion selecting a marginally different config, not the division-to-multiply transformation itself. The strategy of converting avg_pool division to reciprocal multiply to enable FMA fusion is ineffective when the surrounding arithmetic chain is already dominated by L1TEX scoreboard stalls (CPIStall at 56.5% of cycles) rather than FP32 instruction throughput.

> **Applicable when**: NCU FPInstructions rule fires AND the kernel is compute-throughput-bound (Compute SM Throughput > 60%) AND the division operand is a compile-time constant power-of-two (enabling exact IEEE 754 reciprocal). Only apply when L1TEX stall fraction is low (<30% of cycles).

> **Avoid when**: The kernel is dominated by L1TEX scoreboard stalls (CPIStall rule fires with L1TEX stall fraction > 50% of cycles) — in this regime, FP32 instruction throughput is not the bottleneck and FMA fusion yields no measurable benefit. Do not apply when executed instruction count is unchanged before and after (as seen here), confirming the compiler already handles the division efficiently. Do not apply when Compute SM Throughput is below 40% (memory-latency-bound, not compute-bound).

#### ✅ Plan D: `replace_avg_pool_division_with_multiply_to_enable_fma_fusion` (1.079×)

Replacing a constant division (sum / 4.0) in avg_pool2d with an equivalent multiply (sum * 0.25) was predicted to enable FMA fusion and reduce non-fused FP32 instruction count (NCU FPInstructions rule: 123K non-fused vs 17K fused). However, the profiling data shows executed instructions remained identical (1,076,040 before and after), elapsed cycles actually increased slightly (8714 → 8999), and the fused/non-fused instruction ratio was unchanged. The 1.08x wall-clock speedup is therefore not attributable to FMA fusion from the division-to-multiply change; it likely reflects noise or a minor secondary effect from the concurrent autotune config pruning (removing half the configs). The FDIV→FMUL substitution alone did not measurably reduce instruction count or stall cycles in this kernel.

> **Applicable when**: NCU FPInstructions rule fires with high non-fused FP32 instruction count; kernel contains a division by a compile-time constant (e.g., /4.0 for avg_pool); the division is the primary source of non-fused instructions; the constant is exactly representable as a reciprocal in IEEE 754 (e.g., 1/4=0.25, 1/8=0.125).

> **Avoid when**: The non-fused instruction count does not decrease after the change (as seen here — executed instructions stayed at 1.076M); when the FPInstructions rule fires but the dominant non-fused instructions come from other sources (e.g., rsqrt chains, weight/bias multiplies) rather than the division; when the kernel is already memory-latency-bound (CPIStall L1TEX > 50% of cycles) and instruction-count reduction is not the bottleneck; when the autotune config set is being simultaneously modified, making it impossible to isolate the effect of the arithmetic change.

---

## Cumulative Progress

```
Baseline:  6560 ns  (1.000×)
Iter 2     5980 ns  (1.097×)  — restore_small_xblock_configs_and_fuse_rsqrt

Final best: 5980 ns  (1.0970×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
     xnumel = 564480
@@ -16,21 +32,21 @@

     x2 = xindex // 4032
     x3 = xindex // 288
     x4 = xindex
-    tmp21 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
-    tmp34 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
+    tmp21 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp23 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp32 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp34 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 256, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -39,24 +55,17 @@

     tmp16 = tmp0 >= tmp3
     tmp17 = tl.full([1], 288, tl.int64)
     tmp18 = tmp0 < tmp17
-    tmp19 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp19 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp16 & xmask, eviction_policy='evict_first', other=0.0)
     tmp20 = tl.where(tmp4, tmp15, tmp19)
     tmp22 = tmp20 - tmp21
     tmp24 = tl.full([1], 1e-05, tl.float32)
     tmp25 = tmp23 + tmp24
-    tmp26 = tl.sqrt_rn(tmp25)
-    tmp27 = tl.full([1], 1, tl.int32)
-    tmp28 = (tmp27 / tmp26)
-    tmp29 = tl.full([1], 1.0, tl.float32)
-    tmp30 = tmp28 * tmp29
+    tmp30 = libdevice.rsqrt(tmp25)
     tmp31 = tmp22 * tmp30
-    tmp33 = tmp31 * tmp32
-    tmp35 = tmp33 + tmp34
+    tmp35 = libdevice.fma(tmp31, tmp32, tmp34)
     tmp36 = tl.full([1], 0, tl.int32)
     tmp37 = tl.maximum(tmp36, tmp35)
-    tl.store(in_out_ptr0 + (x4), tmp37, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp37, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -68,7 +77,7 @@

     arg_6 = inputs[6]
     xnumel = 564480
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 564480, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 564480)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
