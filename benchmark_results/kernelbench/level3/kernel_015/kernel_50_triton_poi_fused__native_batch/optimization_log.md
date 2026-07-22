# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10820 ns — **Best latency**: 9310 ns (**1.1622×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_batchnorm_epilogue_arithmetic_to_fma_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_50_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6860,1,1), Waves/SM: 3.25
- **Registers/thread**: 32
- **Duration**: 10820 ns
- **SM Throughput**: 59.57 % | **Memory**: 21.44 % | **DRAM**: 18.01 %
- **Achieved Occupancy**: 80.19 %
- **IPC Active**: 2.97 inst/cycle | **No Eligible**: 22.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_eliminate_partial_wave | 1.002× | 0.405× | ✓ | **✓ best** |
| B | increase_xblock_eliminate_partial_wave | 0.962× | 0.937× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_eliminate_tail_wave | 0.948× | 0.216× | ✓ |  |
| D | increase_xblock_eliminate_partial_wave_tail | 0.948× | 0.937× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_e... | Δ | increase_xblock_f... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10820 ns | 26690 ns | +15870 ns (+146.7%) | 11550 ns | +730 ns (+6.7%) | 49980 ns | +39160 ns (+361.9%) | 11550 ns | +730 ns (+6.7%) |
| SM Throughput (%) | 59.57 % | 72.85 % | +13.28 pp | 66.57 % | +7.00 pp | 76.98 % | +17.41 pp | 66.63 % | +7.06 pp |
| Memory Throughput (%) | 21.44 % | 18.64 % | -2.80 pp | 24.47 % | +3.03 pp | 18.29 % | -3.15 pp | 24.55 % | +3.11 pp |
| DRAM Throughput (%) | 18.01 % | 7.29 % | -10.72 pp | 16.88 % | -1.13 pp | 3.89 % | -14.12 pp | 16.88 % | -1.13 pp |
| Achieved Occupancy (%) | 80.19 % | 66.44 % | -13.75 pp | 82.47 % | +2.28 pp | 67.15 % | -13.04 pp | 82.35 % | +2.16 pp |
| Registers/Thread | 32.00 register/thread | 38.00 register/thread | +6 | 32.00 register/thread | +0 | 38.00 register/thread | +6 | 32.00 register/thread | +0 |
| Waves/SM | 3.25 | 8.66 | +5.41 | 3.25 | +0.00 | 17.32 | +14.07 | 3.25 | +0.00 |
| No Eligible (%) | 22.25 % | 20.96 % | -1.29 pp | 18.69 % | -3.56 pp | 19.10 % | -3.15 pp | 18.89 % | -3.36 pp |
| IPC Active | 2.97 inst/cycle | 3.13 inst/cycle | +0.16 | 3.19 inst/cycle | +0.22 | 3.23 inst/cycle | +0.26 | 3.22 inst/cycle | +0.25 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_scheduler_stall_latency | 1.137× | 0.577× | ✓ |  |
| B | fix_autotune_configs_to_include_xblock128_and_explore_num_warps | 0.982× | 1.025× | ✓ |  |
| C | reduce_register_pressure_via_smaller_xblock_and_fewer_warps | 1.000× | 0.985× | ✓ |  |
| D | reduce_xblock_to_restore_register_limited_occupancy | 1.160× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_autotune_conf... | Δ | reduce_register_p... | Δ | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10820 ns | 18750 ns | +7930 ns (+73.3%) | 10560 ns | -260 ns (-2.4%) | 10980 ns | +160 ns (+1.5%) | 10880 ns | +60 ns (+0.6%) |
| SM Throughput (%) | 59.57 % | 69.21 % | +9.64 pp | 61.07 % | +1.50 pp | 60.62 % | +1.05 pp | 60.82 % | +1.25 pp |
| Memory Throughput (%) | 21.44 % | 23.98 % | +2.54 pp | 21.97 % | +0.53 pp | 21.81 % | +0.37 pp | 21.88 % | +0.44 pp |
| DRAM Throughput (%) | 18.01 % | 10.38 % | -7.63 pp | 18.49 % | +0.48 pp | 17.77 % | -0.24 pp | 17.93 % | -0.08 pp |
| Achieved Occupancy (%) | 80.19 % | 84.58 % | +4.39 pp | 80.41 % | +0.22 pp | 80.23 % | +0.04 pp | 80.15 % | -0.04 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 3.25 | 6.50 | +3.25 | 3.25 | +0.00 | 3.25 | +0.00 | 3.25 | +0.00 |
| No Eligible (%) | 22.25 % | 21.01 % | -1.24 pp | 22.42 % | +0.17 pp | 22.19 % | -0.06 pp | 22.79 % | +0.54 pp |
| IPC Active | 2.97 inst/cycle | 3.15 inst/cycle | +0.18 | 3.07 inst/cycle | +0.10 | 3.06 inst/cycle | +0.09 | 3.06 inst/cycle | +0.09 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_non_fused_fp32_instructions_into_fma_in_batchnorm_epilogue | 1.081× | 1.190× | ✓ |  |
| B | fuse_batchnorm_epilogue_arithmetic_to_fma_instructions | 1.161× | 1.162× | ✓ | **✓ best** |
| C | replace_sqrt_div_with_rsqrt_to_enable_fma_fusion | 1.100× | 1.186× | ✓ |  |
| D | fuse_fp32_instructions_via_rsqrt_and_fma | 1.078× | 1.162× | ✓ |  |
| E | add_triton_autotune | 0.984× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_non_fused_fp... | Δ | fuse_batchnorm_ep... | Δ | replace_sqrt_div_... | Δ | fuse_fp32_instruc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10820 ns | 9090 ns | -1730 ns (-16.0%) | 9310 ns | -1510 ns (-14.0%) | 9120 ns | -1700 ns (-15.7%) | 9310 ns | -1510 ns (-14.0%) | 10780 ns | -40 ns (-0.4%) |
| SM Throughput (%) | 59.57 % | 56.76 % | -2.81 pp | 56.83 % | -2.74 pp | 57.46 % | -2.11 pp | 56.87 % | -2.70 pp | 61.50 % | +1.93 pp |
| Memory Throughput (%) | 21.44 % | 25.56 % | +4.12 pp | 25.60 % | +4.16 pp | 25.88 % | +4.44 pp | 25.61 % | +4.17 pp | 22.12 % | +0.68 pp |
| DRAM Throughput (%) | 18.01 % | 21.52 % | +3.51 pp | 20.94 % | +2.93 pp | 21.32 % | +3.31 pp | 20.89 % | +2.88 pp | 18.09 % | +0.08 pp |
| Achieved Occupancy (%) | 80.19 % | 77.56 % | -2.63 pp | 77.66 % | -2.53 pp | 77.92 % | -2.27 pp | 78.44 % | -1.75 pp | 80.50 % | +0.31 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 3.25 | 3.25 | +0.00 | 3.25 | +0.00 | 3.25 | +0.00 | 3.25 | +0.00 | 3.25 | +0.00 |
| No Eligible (%) | 22.25 % | 24.50 % | +2.25 pp | 24.56 % | +2.31 pp | 24.59 % | +2.34 pp | 24.43 % | +2.18 pp | 22.48 % | +0.23 pp |
| IPC Active | 2.97 inst/cycle | 2.96 inst/cycle | -0.01 | 2.97 inst/cycle | +0.00 | 2.97 inst/cycle | +0.00 | 2.99 inst/cycle | +0.02 | 3.08 inst/cycle | +0.11 |


### Key Learnings

#### ✅ Plan B: `fuse_batchnorm_epilogue_arithmetic_to_fma_instructions` (1.162×)

When the NCU FPInstructions rule fires with a high fraction of non-fused FP32 instructions (>75%), replacing the batch-norm normalize-scale-bias chain (sqrt+rcp+mul+mul+add) with tl.math.rsqrt and tl.math.fma reduces total executed instructions by ~19% (5.1M→4.1M) and kernel duration by ~12% (10.56→9.31 µs). The key causal chain is: removing the redundant int32/float32 type-conversion division (int32(1)/sqrt) and the no-op ×1.0 multiply eliminates divergent branches (274K→55K branch instructions, avg divergent branches 51.97→0.0), improving average active threads per warp from 30.25 to 32.0 (full warp utilization). The autotune config change (replacing num_warps=16 with XBLOCK=64/num_warps=4) also contributed by avoiding the previously-observed regression from high num_warps configs.

> **Applicable when**: NCU FPInstructions rule fires with >70% non-fused FP32 fraction; kernel contains batch-norm epilogue arithmetic (normalize-scale-bias-relu chain); profiling shows divergent branches and avg active threads per warp below 32 (indicating predicated-off threads from type-conversion branches); kernel is compute-bound (Compute SOL > Memory SOL).

> **Avoid when**: Kernel requires strict IEEE 754 double-rounding semantics (tl.math.fma differs from (a*b)+c by up to 1 ULP); variance is not guaranteed non-negative (rsqrt is undefined for negative inputs); the non-fused instruction count is already low (<30% of total FP32 instructions, meaning the FPInstructions rule is not triggered).

#### ✅ Plan D: `fuse_fp32_instructions_via_rsqrt_and_fma` (1.162×)

When NCU's FPInstructions rule identifies a low fused-FP32 fraction (e.g., <25% fused), replacing a multi-instruction normalization sequence (sqrt_rn + integer reciprocal + redundant float multiply) with libdevice.rsqrt() and replacing separate multiply+add with libdevice.fma() reduces total executed instructions by ~19% (5.1M → 4.1M) and kernel duration by ~12% (10.56 µs → 9.31 µs). Simultaneously, replacing division by a power-of-2 constant with multiplication (e.g., /4.0 → *0.25) eliminates one non-fused instruction with no precision loss. Pinning the autotune to a single safe config (XBLOCK=128, num_warps=4) prevents the autotuner from selecting regressive larger-tile configs, ensuring the instruction-count reduction is the dominant effect.

> **Applicable when**: NCU FPInstructions rule fires with a low fused-FP32 fraction (<25%); kernel contains a batch-norm or layer-norm normalization sequence using sqrt+reciprocal; kernel contains division by a power-of-2 constant; kernel is compute-bound or near the compute/memory crossover point (Compute SOL >50%); XBLOCK is already pinned to a safe value (no risk of autotune selecting a regressive larger tile).

> **Avoid when**: The compiler already auto-fuses the sqrt+reciprocal chain in PTX (verify with actual instruction counts before and after); kernel is purely memory-bound (Memory SOL >80%) where instruction count reduction has negligible effect on duration; libdevice.rsqrt() is not available in the target framework or compute capability; FMA semantics (single rounding) would violate required numerical precision (e.g., strict IEEE 754 double-rounding required).

---

## Iteration 4

**Best so far**: 1.1622× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_large_tensor_eviction_policy_to_evict_first_for_cache_locality | 1.079× | 1.159× | ✓ |  |
| C | change_eviction_policy_streaming_loads_to_evict_first | 1.091× | 1.159× | ✓ |  |
| D | differentiate_eviction_policies_streaming_vs_broadcast_loads | 1.059× | 1.166× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_large_tens... | Δ | change_eviction_p... | Δ | differentiate_evi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10820 ns | 9340 ns | -1480 ns (-13.7%) | 9340 ns | -1480 ns (-13.7%) | 9280 ns | -1540 ns (-14.2%) |
| SM Throughput (%) | 59.57 % | 57.52 % | -2.05 pp | 57.11 % | -2.46 pp | 56.85 % | -2.72 pp |
| Memory Throughput (%) | 21.44 % | 25.90 % | +4.46 pp | 25.71 % | +4.27 pp | 25.61 % | +4.17 pp |
| DRAM Throughput (%) | 18.01 % | 20.89 % | +2.88 pp | 20.93 % | +2.92 pp | 20.96 % | +2.95 pp |
| Achieved Occupancy (%) | 80.19 % | 78.35 % | -1.84 pp | 76.42 % | -3.77 pp | 77.71 % | -2.48 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 3.25 | 3.25 | +0.00 | 3.25 | +0.00 | 3.25 | +0.00 |
| No Eligible (%) | 22.25 % | 24.46 % | +2.21 pp | 26.53 % | +4.28 pp | 26.17 % | +3.92 pp |
| IPC Active | 2.97 inst/cycle | 2.99 inst/cycle | +0.02 | 2.87 inst/cycle | -0.10 | 2.96 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan A: `change_large_tensor_eviction_policy_to_evict_first_for_cache_locality` (1.159×)

Switching large streaming input tensor loads from eviction_policy='evict_last' to 'evict_first' (while keeping small parameter vectors on 'evict_last') did NOT improve L1/TEX hit rate (remained at 48.36%), L2 hit rate slightly worsened (32.41% → 31.61%), and kernel duration actually increased slightly (9.09 µs → 9.34 µs) in the NCU profiling trace. Despite the 1.16x reported end-to-end speedup, the per-kernel profiling metrics show no cache improvement from this eviction policy change — the L1 and L2 hit rates were unchanged or marginally worse, and warp cycles per issued instruction increased (16.62 → 17.24). The H200's large 50MB L2 cache appears to already absorb the parameter vector traffic regardless of eviction policy, making this hint ineffective for this kernel class.

> **Applicable when**: Kernel loads a mix of large streaming tensors (read-once, no reuse) and small parameter vectors (reused across many thread blocks); L1/TEX hit rate is moderate (40-60%) and L2 hit rate is low (<40%); kernel is latency-bound with CPIStall >40%; the small parameter vectors are small enough to fit in L1 (< L1 capacity per SM).

> **Avoid when**: The GPU has a very large L2 cache (e.g., H200 with 50MB L2) that already absorbs parameter vector traffic regardless of eviction policy — in this case the hint has no measurable effect on hit rates. Also avoid when the kernel's L1/TEX hit rate does not change after the policy switch (as observed here: 48.36% before and after), indicating the hardware is ignoring or overriding the hint. Do not apply when the end-to-end speedup is driven by factors other than this kernel (e.g., other kernels in the pipeline benefiting from freed cache capacity).

#### ✅ Plan C: `change_eviction_policy_streaming_loads_to_evict_first` (1.159×)

Changing tl.load eviction_policy from 'evict_last' to 'evict_first' on large one-use streaming input tensors (while keeping 'evict_last' on small frequently-reused parameter tensors) did NOT improve L2 or L1 hit rates as predicted — L2 hit rate was essentially unchanged (32.41% → 32.24%) and L1 hit rate was identical (48.36%). Despite the plan's rationale being sound in theory, the profiling data shows no measurable cache benefit, and the kernel duration actually increased slightly (9.09 µs → 9.34 µs). The 1.1585x reported speedup appears to reflect end-to-end measurement variance rather than a genuine kernel-level improvement, as all NCU metrics are essentially flat or marginally worse. The autotune config change (replacing num_warps=16 with XBLOCK=64/num_warps=4 and num_warps=2) was bundled into the same diff and may account for the apparent speedup via autotune selection of a slightly different config.

> **Applicable when**: Kernel mixes large one-use streaming tensors with small frequently-reused parameter tensors; L2 hit rate is low (<40%) and the reused parameters are small enough to fit in L2 but are being evicted by the streaming data; the streaming tensors are accessed exactly once per kernel invocation with no cross-block reuse.

> **Avoid when**: L2 hit rate does not improve after the change (as observed here — 32.41% → 32.24%); the kernel is already at the memory bandwidth ceiling (DRAM throughput >80%); the streaming tensors are small enough that eviction policy has negligible effect on cache pressure; the kernel duration is dominated by compute rather than memory latency (Compute SOL >70%). Do not apply when the kernel is already near its performance floor and measurement variance exceeds the expected gain.

#### ✅ Plan D: `differentiate_eviction_policies_streaming_vs_broadcast_loads` (1.166×)

Switching large streaming tensor loads (read-once, multi-MB) from evict_last to evict_first while keeping small broadcast parameter loads (sub-10KB, reused across all blocks) on evict_last was predicted to improve L2 hit rate by preventing streaming data from evicting resident broadcast parameters. However, the profiling data shows the L2 hit rate did NOT improve (32.41% → 31.81%), memory throughput slightly decreased (1.05 → 1.03 TB/s), and kernel duration actually increased (9.09 → 9.28 µs) in the NCU single-kernel measurement — the 1.17x speedup was measured at the end-to-end benchmark level, not from L2 cache improvement. The NCU metrics are essentially unchanged between before and after, suggesting the speedup may be attributable to the simultaneous autotune config simplification (removing num_warps=8 and num_warps=16 configs, pinning to a single num_warps=4 config) rather than the eviction policy change itself.

> **Applicable when**: Kernel mixes large streaming tensors (>1MB, read-once per output element) with small broadcast parameters (<100KB, reused across all thread blocks); L2 hit rate is low (<40%) despite evict_last on all loads; MemoryL2Compression rule fires; the broadcast parameters are small enough to fit in L2 (<<50MB on H200).

> **Avoid when**: NCU single-kernel profiling shows no L2 hit rate improvement after applying eviction policy differentiation; the kernel duration in NCU actually increases (as seen here: 9.09→9.28µs); the speedup is confounded by simultaneous autotune simplification. Avoid when the kernel is already near the compute roofline (SM throughput ~57%) where memory hierarchy changes have limited impact.

---

## Iteration 5

**Best so far**: 1.1622× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_256_eliminate_partial_wave_and_reduce_scheduler_contention | 1.080× | 0.719× | ✓ |  |
| B | align_xblock_to_channel_width_448_to_eliminate_partial_wave_and_reduce_scheduler_stall | 1.012× | 0.904× | ✓ |  |
| C | align_grid_to_full_waves_via_xblock256_and_reduce_cpi_stall_via_num_warps_8 | 1.002× | 0.661× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | align_xblock_to_c... | Δ | align_grid_to_ful... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10820 ns | 15040 ns | +4220 ns (+39.0%) | 11970 ns | +1150 ns (+10.6%) | 16380 ns | +5560 ns (+51.4%) |
| SM Throughput (%) | 59.57 % | 69.87 % | +10.30 pp | 59.54 % | -0.03 pp | 70.12 % | +10.55 pp |
| Memory Throughput (%) | 21.44 % | 29.86 % | +8.42 pp | 25.52 % | +4.08 pp | 30.06 % | +8.62 pp |
| DRAM Throughput (%) | 18.01 % | 12.94 % | -5.07 pp | 16.27 % | -1.74 pp | 11.88 % | -6.13 pp |
| Achieved Occupancy (%) | 80.19 % | 83.75 % | +3.56 pp | 46.20 % | -33.99 pp | 63.18 % | -17.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 56.00 register/thread | +24 | 40.00 register/thread | +8 |
| Waves/SM | 3.25 | 6.50 | +3.25 | 5.77 | +2.52 | 8.66 | +5.41 |
| No Eligible (%) | 22.25 % | 19.00 % | -3.25 pp | 26.70 % | +4.45 pp | 18.60 % | -3.65 pp |
| IPC Active | 2.97 inst/cycle | 3.23 inst/cycle | +0.26 | 2.93 inst/cycle | -0.04 | 3.25 inst/cycle | +0.28 |

---

## Cumulative Progress

```
Baseline:  10820 ns  (1.000×)
Iter 3     9310 ns  (1.162×)  — fuse_batchnorm_epilogue_arithmetic_to_fma_instructions

Final best: 9310 ns  (1.1622×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
     xnumel = 878080
@@ -16,55 +23,54 @@

     x2 = xindex // 6272
     x3 = xindex // 448
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
-    tmp12 = tl.full([1], 4.0, tl.float32)
-    tmp13 = (tmp11 / tmp12)
+    tmp13 = tmp11 * 0.25
     tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
     tmp15 = tl.where(tmp4, tmp13, tmp14)
     tmp16 = tmp0 >= tmp3
     tmp17 = tl.full([1], 288, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_first', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 320, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp24 & xmask, eviction_policy='evict_first', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 352, tl.int64)
     tmp28 = tmp0 < tmp27
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), mask=tmp29 & xmask, eviction_policy='evict_first', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1], 384, tl.int64)
     tmp33 = tmp0 < tmp32
     tmp34 = tmp31 & tmp33
-    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), mask=tmp34 & xmask, eviction_policy='evict_first', other=0.0)
     tmp36 = tmp0 >= tmp32
     tmp37 = tl.full([1], 416, tl.int64)
     tmp38 = tmp0 < tmp37
     tmp39 = tmp36 & tmp38
-    tmp40 = tl.load(in_ptr5 + (32*x3 + ((-384) + x0)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp40 = tl.load(in_ptr5 + (32*x3 + ((-384) + x0)), mask=tmp39 & xmask, eviction_policy='evict_first', other=0.0)
     tmp41 = tmp0 >= tmp37
     tmp42 = tl.full([1], 448, tl.int64)
     tmp43 = tmp0 < tmp42
-    tmp44 = tl.load(in_ptr6 + (32*x3 + ((-416) + x0)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp44 = tl.load(in_ptr6 + (32*x3 + ((-416) + x0)), mask=tmp41 & xmask, eviction_policy='evict_first', other=0.0)
     tmp45 = tl.where(tmp39, tmp40, tmp44)
     tmp46 = tl.where(tmp34, tmp35, tmp45)
     tmp47 = tl.where(tmp29, tmp30, tmp46)
@@ -74,19 +80,12 @@

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
+    inv_std = libdevice.rsqrt(tmp55)
+    tmp61 = tmp52 * inv_std
+    tmp65 = libdevice.fma(tmp61, tmp62, tmp64)
     tmp66 = tl.full([1], 0, tl.int32)
     tmp67 = tl.maximum(tmp66, tmp65)
-    tl.store(in_out_ptr0 + (x4), tmp67, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp67, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -103,7 +102,7 @@

     arg_11 = inputs[11]
     xnumel = 878080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, 878080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, 878080)
     return arg_11
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
