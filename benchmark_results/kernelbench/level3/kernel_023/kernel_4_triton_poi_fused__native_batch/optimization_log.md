# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24800 ns — **Best latency**: 24540 ns (**1.0106×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_broken_autotune_launch_and_remove_dead_parameter_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (36000,1,1), Waves/SM: 17.05
- **Registers/thread**: 17
- **Duration**: 24800 ns
- **SM Throughput**: 54.80 % | **Memory**: 23.92 % | **DRAM**: 15.18 %
- **Achieved Occupancy**: 55.58 %
- **IPC Active**: 2.42 inst/cycle | **No Eligible**: 39.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.734× | 0.712× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.909× | 0.908× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.726× | 0.712× | ✓ |  |
| D | autotune_xblock_for_occupancy_and_latency_hiding | 0.998× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | add_autotune_with... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24800 ns | 34850 ns | +10050 ns (+40.5%) | 27300 ns | +2500 ns (+10.1%) | 34850 ns | +10050 ns (+40.5%) | 24770 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 54.80 % | 83.71 % | +28.91 pp | 75.77 % | +20.97 pp | 83.52 % | +28.72 pp | 64.41 % | +9.61 pp |
| Memory Throughput (%) | 23.92 % | 25.25 % | +1.33 pp | 29.33 % | +5.41 pp | 25.13 % | +1.21 pp | 24.17 % | +0.25 pp |
| DRAM Throughput (%) | 15.18 % | 10.80 % | -4.38 pp | 13.78 % | -1.40 pp | 10.80 % | -4.38 pp | 15.19 % | +0.01 pp |
| Achieved Occupancy (%) | 55.58 % | 87.88 % | +32.30 pp | 64.61 % | +9.03 pp | 87.31 % | +31.73 pp | 59.84 % | +4.26 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 |
| No Eligible (%) | 39.25 % | 10.96 % | -28.29 pp | 15.91 % | -23.34 pp | 11.64 % | -27.61 pp | 29.55 % | -9.70 pp |
| IPC Active | 2.42 inst/cycle | 3.54 inst/cycle | +1.12 | 3.37 inst/cycle | +0.95 | 3.54 inst/cycle | +1.12 | 2.84 inst/cycle | +0.42 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.712×)

Adding @triton.autotune with large XBLOCK configs (512–2048) caused registers per thread to double from 17 to 32, and total executed instructions to more than double (10.7M → 22.8M), resulting in a 40% longer kernel duration despite higher achieved occupancy (55.6% → 87.9%). The autotune selected a config that increased register pressure and instruction count far beyond what was needed to hide latency, negating the occupancy benefit. When a pointwise kernel is already compute-bound at moderate occupancy with low register usage, introducing large tile sizes via autotune can trigger register pressure increases and instruction bloat that outweigh the scheduling gains.

> **Applicable when**: Never apply this specific pattern. Avoid using autotune with exclusively large XBLOCK configs (≥512) for pointwise kernels that already have low register usage (≤20 regs/thread) and moderate occupancy (50–60%). The instruction count explosion and register pressure increase will dominate any scheduling benefit.

> **Avoid when**: Do not apply large-XBLOCK autotune when: (1) registers per thread are already low (≤20), suggesting the compiler is not register-limited; (2) the kernel is purely pointwise with no data reuse across elements; (3) the existing grid already provides many waves (>10 waves/SM), meaning occupancy is not the primary bottleneck. In these cases, the autotune overhead and register pressure from larger tiles cause regression.

#### ❌ Plan C: `add_autotune_with_larger_xblock_for_vectorized_loads` (0.712×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) doubled register usage per thread (17→32) and more than doubled the executed instruction count (10.6M→22.8M), causing a 40% slowdown despite improving achieved occupancy (55.6%→87.3%) and scheduler eligibility. The autotune selected a config that increased register pressure enough to double instruction count, indicating the larger block sizes introduced additional index computation or spill/reload overhead that outweighed any vectorization benefit. When the original kernel already has low register count (≤20/thread) and the bottleneck is L1TEX stalls rather than occupancy, adding autotune with large XBLOCK can backfire by selecting configs that inflate register usage and instruction count.

> **Applicable when**: Never — this is a negative example. Do not apply autotune with large XBLOCK when the kernel has low register count (≤20/thread) and the primary bottleneck is L1TEX scoreboard stalls in a pointwise kernel.

> **Avoid when**: Pointwise kernels with low register count (≤20/thread) where L1TEX stalls are the bottleneck; adding large XBLOCK autotune configs risks doubling register usage and instruction count, causing significant slowdown even if occupancy improves. Also: the grid was not updated to use lambda meta, so autotune cannot actually change the grid size — the grid remained fixed at cdiv(xnumel, 128)=36000 regardless of selected XBLOCK, causing incorrect behavior.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_to_hide_l1tex_stalls | 0.993× | 1.000× | ✓ |  |
| B | fix_broken_autotune_grid_and_enable_fma_fusion | 1.001× | 0.998× | ✓ |  |
| C | replace_sqrt_div_with_rsqrt_and_remove_bloated_autotune | 1.007× | 1.011× | ✓ | **✓ best** |
| D | eliminate_integer_division_to_enable_fma_fusion | 0.917× | 0.955× | ✓ |  |
| E | add_triton_autotune | 0.993× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_broken_autotu... | Δ | replace_sqrt_div_... | Δ | eliminate_integer... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24800 ns | 24800 ns | +0 ns (+0.0%) | 24860 ns | +60 ns (+0.2%) | 24540 ns | -260 ns (-1.0%) | 25980 ns | +1180 ns (+4.8%) | 24770 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 54.80 % | 64.58 % | +9.78 pp | 44.44 % | -10.36 pp | 39.30 % | -15.50 pp | 51.85 % | -2.95 pp | 64.62 % | +9.82 pp |
| Memory Throughput (%) | 23.92 % | 24.17 % | +0.25 pp | 32.01 % | +8.09 pp | 23.53 % | -0.39 pp | 34.90 % | +10.98 pp | 24.18 % | +0.26 pp |
| DRAM Throughput (%) | 15.18 % | 15.17 % | -0.01 pp | 15.11 % | -0.07 pp | 15.33 % | +0.15 pp | 14.47 % | -0.71 pp | 15.20 % | +0.02 pp |
| Achieved Occupancy (%) | 55.58 % | 59.65 % | +4.07 pp | 31.69 % | -23.89 pp | 41.43 % | -14.15 pp | 38.18 % | -17.40 pp | 60.46 % | +4.88 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 |
| No Eligible (%) | 39.25 % | 28.95 % | -10.30 pp | 48.43 % | +9.18 pp | 56.15 % | +16.90 pp | 41.69 % | +2.44 pp | 28.58 % | -10.67 pp |
| IPC Active | 2.42 inst/cycle | 2.85 inst/cycle | +0.43 | 2.05 inst/cycle | -0.37 | 1.75 inst/cycle | -0.67 | 2.34 inst/cycle | -0.08 | 2.86 inst/cycle | +0.44 |

---

## Iteration 3

**Best so far**: 1.0106× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 1.009× | 1.011× | ✓ | **✓ best** |
| B | replace_int_div_reciprocal_sqrt_with_rsqrt_for_fma_fusion | 1.009× | 1.009× | ✓ |  |
| C | fix_broken_autotune_grid_and_enable_fma_via_channel_index_rewrite | 1.008× | 1.004× | ✓ |  |
| D | set_evict_first_on_streaming_main_load_to_reduce_l1tex_stalls | 0.918× | 0.953× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | replace_int_div_r... | Δ | fix_broken_autotu... | Δ | set_evict_first_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24800 ns | 24540 ns | -260 ns (-1.0%) | 24580 ns | -220 ns (-0.9%) | 24700 ns | -100 ns (-0.4%) | 26020 ns | +1220 ns (+4.9%) |
| SM Throughput (%) | 54.80 % | 39.37 % | -15.43 pp | 39.36 % | -15.44 pp | 21.47 % | -33.33 pp | 54.60 % | -0.20 pp |
| Memory Throughput (%) | 23.92 % | 23.60 % | -0.32 pp | 23.63 % | -0.29 pp | 21.49 % | -2.43 pp | 34.82 % | +10.90 pp |
| DRAM Throughput (%) | 15.18 % | 15.32 % | +0.14 pp | 15.32 % | +0.14 pp | 15.23 % | +0.05 pp | 14.45 % | -0.73 pp |
| Achieved Occupancy (%) | 55.58 % | 41.60 % | -13.98 pp | 41.71 % | -13.87 pp | 25.46 % | -30.12 pp | 40.28 % | -15.30 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 32.00 register/thread | +15 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 8.52 | -8.53 | 17.05 | +0.00 |
| No Eligible (%) | 39.25 % | 56.18 % | +16.93 pp | 56.24 % | +16.99 pp | 74.41 % | +35.16 pp | 38.18 % | -1.07 pp |
| IPC Active | 2.42 inst/cycle | 1.76 inst/cycle | -0.66 | 1.75 inst/cycle | -0.67 | 0.99 inst/cycle | -1.43 | 2.43 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0106× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_fusion_via_arithmetic_restructuring | 1.007× | 1.011× | ✓ |  |
| B | precompute_batchnorm_scale_and_bias_to_eliminate_per_element_rsqrt | 1.008× | 1.011× | ✓ |  |
| C | increase_warps_within_safe_xblock_range_for_latency_hiding | 1.007× | 1.001× | ✓ |  |
| D | fuse_non_fma_fp32_instructions_via_explicit_fma_intrinsic | 0.910× | 0.951× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_fusi... | Δ | precompute_batchn... | Δ | increase_warps_wi... | Δ | fuse_non_fma_fp32... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24800 ns | 24540 ns | -260 ns (-1.0%) | 24540 ns | -260 ns (-1.0%) | 24770 ns | -30 ns (-0.1%) | 26080 ns | +1280 ns (+5.2%) |
| SM Throughput (%) | 54.80 % | 38.32 % | -16.48 pp | 26.71 % | -28.09 pp | 39.38 % | -15.42 pp | 54.27 % | -0.53 pp |
| Memory Throughput (%) | 23.92 % | 23.34 % | -0.58 pp | 19.03 % | -4.89 pp | 23.57 % | -0.35 pp | 34.75 % | +10.83 pp |
| DRAM Throughput (%) | 15.18 % | 15.31 % | +0.13 pp | 15.33 % | +0.15 pp | 15.17 % | -0.01 pp | 14.41 % | -0.77 pp |
| Achieved Occupancy (%) | 55.58 % | 42.32 % | -13.26 pp | 38.14 % | -17.44 pp | 41.62 % | -13.96 pp | 39.89 % | -15.69 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 |
| No Eligible (%) | 39.25 % | 57.11 % | +17.86 pp | 70.24 % | +30.99 pp | 56.16 % | +16.91 pp | 38.17 % | -1.08 pp |
| IPC Active | 2.42 inst/cycle | 1.70 inst/cycle | -0.72 | 1.19 inst/cycle | -1.23 | 1.75 inst/cycle | -0.67 | 2.43 inst/cycle | +0.01 |

---

## Iteration 5

**Best so far**: 1.0106× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_to_256_threads_for_intra_block_latency_hiding | 1.003× | 1.009× | ✓ |  |
| B | fix_broken_autotune_launch_and_remove_dead_parameter_loads | 1.004× | 1.011× | ✓ |  |
| C | fix_grid_lambda_to_enable_safe_autotune_with_xblock_256 | 1.008× | 1.011× | ✓ |  |
| D | fix_broken_autotune_grid_with_small_xblock_configs_to_reduce_launch_overhead | 1.000× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | fix_broken_autotu... | Δ | fix_grid_lambda_t... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24800 ns | 24580 ns | -220 ns (-0.9%) | 24540 ns | -260 ns (-1.0%) | 24540 ns | -260 ns (-1.0%) | 24640 ns | -160 ns (-0.6%) |
| SM Throughput (%) | 54.80 % | 38.53 % | -16.27 pp | 26.68 % | -28.12 pp | 39.29 % | -15.51 pp | 42.32 % | -12.48 pp |
| Memory Throughput (%) | 23.92 % | 23.48 % | -0.44 pp | 19.03 % | -4.89 pp | 23.54 % | -0.38 pp | 23.61 % | -0.31 pp |
| DRAM Throughput (%) | 15.18 % | 15.30 % | +0.12 pp | 15.33 % | +0.15 pp | 15.32 % | +0.14 pp | 15.26 % | +0.08 pp |
| Achieved Occupancy (%) | 55.58 % | 41.90 % | -13.68 pp | 38.28 % | -17.30 pp | 41.65 % | -13.93 pp | 42.84 % | -12.74 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 17.05 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 | 17.05 | +0.00 |
| No Eligible (%) | 39.25 % | 56.96 % | +17.71 pp | 70.24 % | +30.99 pp | 56.08 % | +16.83 pp | 52.72 % | +13.47 pp |
| IPC Active | 2.42 inst/cycle | 1.72 inst/cycle | -0.70 | 1.19 inst/cycle | -1.23 | 1.75 inst/cycle | -0.67 | 1.87 inst/cycle | -0.55 |

---

## Cumulative Progress

```
Baseline:  24800 ns  (1.000×)
Iter 2     24540 ns  (1.011×)  — replace_sqrt_div_with_rsqrt_and_remove_bloated_autotune

Final best: 24540 ns  (1.0106×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4608000
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 32)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp8 = tl.math.rsqrt(tmp5)
+    tmp_norm_scale = tmp8 * tmp12
+    tmp15 = tl.math.fma(tmp2, tmp_norm_scale, tmp14)
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
