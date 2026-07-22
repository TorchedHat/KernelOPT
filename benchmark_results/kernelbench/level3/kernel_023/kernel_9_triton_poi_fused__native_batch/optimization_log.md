# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 19300 ns — **Best latency**: 19040 ns (**1.0137×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`align_xblock_to_channel_count_to_eliminate_modulo_and_reduce_l1tex_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (27000,1,1), Waves/SM: 12.78
- **Registers/thread**: 18
- **Duration**: 19300 ns
- **SM Throughput**: 61.94 % | **Memory**: 23.53 % | **DRAM**: 14.63 %
- **Achieved Occupancy**: 61.63 %
- **IPC Active**: 2.79 inst/cycle | **No Eligible**: 29.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size | 1.009× | 0.998× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 1.001× | 1.002× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads | 0.738× | 0.711× | ✓ |  |
| D | increase_xblock_and_autotune_for_occupancy_and_memory_latency_hiding | 0.738× | 0.709× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19300 ns | 19330 ns | +30 ns (+0.2%) | 19260 ns | -40 ns (-0.2%) | 27140 ns | +7840 ns (+40.6%) | 27230 ns | +7930 ns (+41.1%) |
| SM Throughput (%) | 61.94 % | 61.62 % | -0.32 pp | 61.89 % | -0.05 pp | 79.66 % | +17.72 pp | 80.05 % | +18.11 pp |
| Memory Throughput (%) | 23.53 % | 23.39 % | -0.14 pp | 23.51 % | -0.02 pp | 23.87 % | +0.34 pp | 24.04 % | +0.51 pp |
| DRAM Throughput (%) | 14.63 % | 14.59 % | -0.04 pp | 14.65 % | +0.02 pp | 10.38 % | -4.25 pp | 10.34 % | -4.29 pp |
| Achieved Occupancy (%) | 61.63 % | 61.54 % | -0.09 pp | 60.97 % | -0.66 pp | 86.66 % | +25.03 pp | 86.97 % | +25.34 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.52 % | 30.08 % | +0.56 pp | 29.54 % | +0.02 pp | 13.52 % | -16.00 pp | 13.42 % | -16.10 pp |
| IPC Active | 2.79 inst/cycle | 2.80 inst/cycle | +0.01 | 2.81 inst/cycle | +0.02 | 3.45 inst/cycle | +0.66 | 3.43 inst/cycle | +0.64 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_with_autotune_for_vectorized_loads` (0.711×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused registers per thread to jump from 18 to 32, nearly doubling total executed instructions (9.3M → 17.0M) and increasing kernel duration from 19.3 µs to 27.14 µs despite improving achieved occupancy (61.6% → 86.7%). The instruction count explosion indicates the compiler unrolled more aggressively or generated additional spill/fill code at larger tile sizes, overwhelming any benefit from wider memory transactions. When the baseline kernel already has low register usage and the bottleneck is L1TEX stalls, increasing XBLOCK without controlling register pressure via `num_stages` or `maxnreg` can cause more harm than good.

> **Applicable when**: Never apply this pattern as described. The autotune approach with large XBLOCK configs is harmful when the baseline kernel has low register count (≤20 registers/thread) and the grid is already static (not a lambda). The static grid does not adapt to the autotuned XBLOCK, so the kernel may run with a mismatched grid.

> **Avoid when**: Do not increase XBLOCK via autotune when: (1) baseline registers per thread are already low (≤20) and larger tiles cause register count to nearly double; (2) the grid is not updated to a lambda that reads meta['XBLOCK'], causing a mismatch between grid size and tile size; (3) the instruction count increase from unrolling outweighs memory transaction savings. Also avoid when the kernel is already compute-bound (Compute SM Throughput > 60%) rather than purely memory-latency-bound.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_for_latency_hiding | 1.000× | 0.610× | ✓ |  |
| B | increase_num_warps_and_simplify_reciprocal_to_hide_l1tex_latency | 1.009× | 1.002× | ✓ | **✓ best** |
| C | fix_int32_reciprocal_to_enable_fma_fusion | 0.746× | 0.711× | ✓ |  |
| D | reduce_register_pressure_with_maxnreg_and_rsqrt_fusion | 1.007× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_int32_recipro... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19300 ns | 31620 ns | +12320 ns (+63.8%) | 19260 ns | -40 ns (-0.2%) | 27140 ns | +7840 ns (+40.6%) | 19070 ns | -230 ns (-1.2%) |
| SM Throughput (%) | 61.94 % | 75.84 % | +13.90 pp | 61.43 % | -0.51 pp | 80.12 % | +18.18 pp | 36.49 % | -25.45 pp |
| Memory Throughput (%) | 23.53 % | 25.38 % | +1.85 pp | 23.32 % | -0.21 pp | 24.04 % | +0.51 pp | 23.05 % | -0.48 pp |
| DRAM Throughput (%) | 14.63 % | 8.91 % | -5.72 pp | 14.74 % | +0.11 pp | 10.38 % | -4.25 pp | 14.79 % | +0.16 pp |
| Achieved Occupancy (%) | 61.63 % | 81.20 % | +19.57 pp | 60.78 % | -0.85 pp | 87.09 % | +25.46 pp | 42.45 % | -19.18 pp |
| Registers/Thread | 18.00 register/thread | 21.00 register/thread | +3 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 12.78 | 25.57 | +12.79 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.52 % | 19.34 % | -10.18 pp | 29.86 % | +0.34 pp | 13.37 % | -16.15 pp | 57.92 % | +28.40 pp |
| IPC Active | 2.79 inst/cycle | 3.23 inst/cycle | +0.44 | 2.80 inst/cycle | +0.01 | 3.43 inst/cycle | +0.64 | 1.69 inst/cycle | -1.10 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_at_fixed_xblock_for_latency_hiding` (0.610×)

Increasing num_warps from 4 to 8 at XBLOCK=128 (and allowing XBLOCK=256 configs) caused the autotune to select XBLOCK=256 with 256 threads/block, which doubled executed instructions (9.3M → 18.7M) and increased kernel duration by 64% (19.26 µs → 31.62 µs). Although eligible warps per scheduler improved (1.84 → 3.75) and achieved occupancy rose (61.0% → 81.2%), the instruction count explosion from the larger tile size overwhelmed any latency-hiding benefit. Even though registers per thread only rose modestly (18 → 21), the doubling of total threads (3.46M → 6.91M) and waves per SM (12.78 → 25.57) indicates the autotune selected the wider XBLOCK=256 config, not the intended XBLOCK=128/num_warps=8 config, causing the regression.

> **Applicable when**: Never apply this pattern on short-duration pointwise kernels (<25 µs) where the autotune config space includes XBLOCK values larger than the baseline, even if the intent is only to increase num_warps at the same XBLOCK. The autotune will select the wider tile if it appears faster during warmup, causing instruction count to scale with XBLOCK.

> **Avoid when**: Do not add larger XBLOCK configs to the autotune search space when: (1) the baseline kernel is already short-duration (<25 µs) and compute-bound (Compute SM Throughput >60%); (2) the instruction count is expected to scale linearly or super-linearly with XBLOCK due to pointwise unrolling; (3) the goal is only to increase num_warps without changing tile size — in that case, restrict autotune configs strictly to the baseline XBLOCK value only. Also avoid when waves per SM are already >10, as additional occupancy gains from more warps per block yield diminishing returns.

---

## Iteration 3

**Best so far**: 1.0021× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_software_pipelining_with_streaming_eviction | 1.008× | 1.012× | ✓ |  |
| B | fuse_sqrt_reciprocal_to_rsqrt_for_fma_instruction_reduction | 1.019× | 1.005× | ✓ | **✓ best** |
| C | fuse_sqrt_reciprocal_to_rsqrt_with_maxnreg_cap | 0.930× | 0.938× | ✓ |  |
| D | revert_xblock256_and_fuse_fp32_reciprocal_for_fma | 1.009× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_so... | Δ | fuse_sqrt_recipro... | Δ | fuse_sqrt_recipro... | Δ | revert_xblock256_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19300 ns | 19070 ns | -230 ns (-1.2%) | 19200 ns | -100 ns (-0.5%) | 20580 ns | +1280 ns (+6.6%) | 19300 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 61.94 % | 39.56 % | -22.38 pp | 36.91 % | -25.03 pp | 48.66 % | -13.28 pp | 61.36 % | -0.58 pp |
| Memory Throughput (%) | 23.53 % | 23.12 % | -0.41 pp | 22.82 % | -0.71 pp | 33.52 % | +9.99 pp | 23.29 % | -0.24 pp |
| DRAM Throughput (%) | 14.63 % | 14.79 % | +0.16 pp | 14.68 % | +0.05 pp | 13.70 % | -0.93 pp | 14.61 % | -0.02 pp |
| Achieved Occupancy (%) | 61.63 % | 43.00 % | -18.63 pp | 42.76 % | -18.87 pp | 42.78 % | -18.85 pp | 61.53 % | -0.10 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 24.00 register/thread | +6 | 18.00 register/thread | +0 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.52 % | 55.03 % | +25.51 pp | 57.45 % | +27.93 pp | 43.86 % | +14.34 pp | 29.81 % | +0.29 pp |
| IPC Active | 2.79 inst/cycle | 1.81 inst/cycle | -0.98 | 1.72 inst/cycle | -1.07 | 2.25 inst/cycle | -0.54 | 2.79 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0052× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_restrict_autotune_to_small_xblock_for_coalesced_channel_last_access | 1.009× | 1.010× | ✓ |  |
| B | replace_max_min_clamp_with_tl_clamp_and_streaming_eviction_on_main_load | 1.018× | 1.014× | ✓ |  |
| C | evict_first_streaming_load_to_free_l1_for_channel_vectors | 1.007× | 1.000× | ✓ |  |
| D | fix_static_grid_to_enable_autotune_lambda | 0.922× | 0.935× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | replace_max_min_c... | Δ | evict_first_strea... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19300 ns | 19100 ns | -200 ns (-1.0%) | 19040 ns | -260 ns (-1.3%) | 19300 ns | +0 ns (+0.0%) | 20640 ns | +1340 ns (+6.9%) |
| SM Throughput (%) | 61.94 % | 39.58 % | -22.36 pp | 41.12 % | -20.82 pp | 68.92 % | +6.98 pp | 48.65 % | -13.29 pp |
| Memory Throughput (%) | 23.53 % | 23.12 % | -0.41 pp | 23.24 % | -0.29 pp | 23.42 % | -0.11 pp | 33.59 % | +10.06 pp |
| DRAM Throughput (%) | 14.63 % | 14.73 % | +0.10 pp | 14.80 % | +0.17 pp | 14.61 % | -0.02 pp | 13.67 % | -0.96 pp |
| Achieved Occupancy (%) | 61.63 % | 43.32 % | -18.31 pp | 43.51 % | -18.12 pp | 66.68 % | +5.05 pp | 42.53 % | -19.10 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 24.00 register/thread | +6 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.52 % | 54.74 % | +25.22 pp | 52.74 % | +23.22 pp | 21.80 % | -7.72 pp | 43.71 % | +14.19 pp |
| IPC Active | 2.79 inst/cycle | 1.82 inst/cycle | -0.97 | 1.88 inst/cycle | -0.91 | 3.10 inst/cycle | +0.31 | 2.22 inst/cycle | -0.57 |

---

## Iteration 5

**Best so far**: 1.0052× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_xblock_to_channel_count_to_eliminate_modulo_and_reduce_l1tex_stalls | 1.029× | 1.014× | ✓ | **✓ best** |
| B | remove_maxnreg_cap_and_restore_occupancy_with_num_stages_sweep | 1.009× | 1.005× | ✓ |  |
| C | fuse_sqrt_reciprocal_to_rsqrt_at_fixed_xblock128_only | 1.021× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_xblock_to_c... | Δ | remove_maxnreg_ca... | Δ | fuse_sqrt_recipro... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19300 ns | 19040 ns | -260 ns (-1.3%) | 19200 ns | -100 ns (-0.5%) | 19140 ns | -160 ns (-0.8%) |
| SM Throughput (%) | 61.94 % | 39.14 % | -22.80 pp | 40.32 % | -21.62 pp | 40.35 % | -21.59 pp |
| Memory Throughput (%) | 23.53 % | 19.06 % | -4.47 pp | 23.17 % | -0.36 pp | 23.18 % | -0.35 pp |
| DRAM Throughput (%) | 14.63 % | 3.71 % | -10.92 pp | 14.68 % | +0.05 pp | 14.73 % | +0.10 pp |
| Achieved Occupancy (%) | 61.63 % | 42.33 % | -19.30 pp | 43.32 % | -18.31 pp | 43.47 % | -18.16 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 12.78 | 12.78 | +0.00 | 12.78 | +0.00 | 12.78 | +0.00 |
| No Eligible (%) | 29.52 % | 54.24 % | +24.72 pp | 53.73 % | +24.21 pp | 53.98 % | +24.46 pp |
| IPC Active | 2.79 inst/cycle | 1.82 inst/cycle | -0.97 | 1.85 inst/cycle | -0.94 | 1.85 inst/cycle | -0.94 |

---

## Cumulative Progress

```
Baseline:  19300 ns  (1.000×)
Iter 2     19260 ns  (1.002×)  — increase_num_warps_and_simplify_reciprocal_to_hide_l1tex_latency
Iter 3     19200 ns  (1.005×)  — fuse_sqrt_reciprocal_to_rsqrt_for_fma_instruction_reduction
Iter 5     19040 ns  (1.014×)  — align_xblock_to_channel_count_to_eliminate_modulo_and_reduce_l1tex_stalls

Final best: 19040 ns  (1.0137×)
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
+        triton.Config({'XBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 32}, num_warps=2),
+        triton.Config({'XBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 64}, num_warps=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3456000
@@ -13,7 +25,7 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 96)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
@@ -21,21 +33,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
+    tmp19 = tl.clamp(tmp15, 0.0, 6.0)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +48,7 @@

     arg_4 = inputs[4]
     xnumel = 3456000
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3456000, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3456000)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
