# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24350 ns — **Best latency**: 24160 ns (**1.0079×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`apply_evict_first_on_streaming_tensor_to_reduce_l1tex_stall`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_12_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (35280,1,1), Waves/SM: 16.70
- **Registers/thread**: 18
- **Duration**: 24350 ns
- **SM Throughput**: 62.92 % | **Memory**: 25.24 % | **DRAM**: 15.24 %
- **Achieved Occupancy**: 59.42 %
- **IPC Active**: 2.79 inst/cycle | **No Eligible**: 30.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_occupancy_and_cache_reuse | 0.998× | 0.206× | ✓ |  |
| B | autotune_xblock_and_fuse_rsqrt | 1.007× | 0.690× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_better_occupancy | 0.727× | 0.703× | ✓ |  |
| D | increase_xblock_with_autotune_for_occupancy_and_latency_hiding | 0.990× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24350 ns | 118340 ns | +93990 ns (+386.0%) | 35300 ns | +10950 ns (+45.0%) | 34620 ns | +10270 ns (+42.2%) | 24350 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 62.92 % | 85.25 % | +22.33 pp | 72.41 % | +9.49 pp | 81.45 % | +18.53 pp | 63.47 % | +0.55 pp |
| Memory Throughput (%) | 25.24 % | 18.18 % | -7.06 pp | 38.36 % | +13.12 pp | 25.79 % | +0.55 pp | 25.45 % | +0.21 pp |
| DRAM Throughput (%) | 15.24 % | 3.13 % | -12.11 pp | 10.53 % | -4.71 pp | 10.69 % | -4.55 pp | 15.29 % | +0.05 pp |
| Achieved Occupancy (%) | 59.42 % | 55.76 % | -3.66 pp | 79.40 % | +19.98 pp | 87.81 % | +28.39 pp | 59.66 % | +0.24 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 53.45 | +36.75 | 33.41 | +16.71 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 30.03 % | 13.07 % | -16.96 pp | 23.16 % | -6.87 pp | 13.35 % | -16.68 pp | 30.20 % | +0.17 pp |
| IPC Active | 2.79 inst/cycle | 3.47 inst/cycle | +0.68 | 3.05 inst/cycle | +0.26 | 3.45 inst/cycle | +0.66 | 2.81 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_int_div_sqrt_with_rsqrt_for_fma_fusion | 0.998× | 1.007× | ✓ |  |
| B | fuse_reciprocal_sqrt_to_rsqrt_and_fix_autotune_grid | 0.997× | 1.009× | ✓ |  |
| C | fuse_batchnorm_scale_bias_into_fma_instruction | 0.999× | 0.689× | ✓ |  |
| D | fix_broken_autotune_grid_and_fuse_rsqrt_instruction | 1.009× | 1.008× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.991× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_int_div_s... | Δ | fuse_reciprocal_s... | Δ | fuse_batchnorm_sc... | Δ | fix_broken_autotu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24350 ns | 24190 ns | -160 ns (-0.7%) | 24130 ns | -220 ns (-0.9%) | 35330 ns | +10980 ns (+45.1%) | 24160 ns | -190 ns (-0.8%) | 24450 ns | +100 ns (+0.4%) |
| SM Throughput (%) | 62.92 % | 38.47 % | -24.45 pp | 38.36 % | -24.56 pp | 72.41 % | +9.49 pp | 38.19 % | -24.73 pp | 63.17 % | +0.25 pp |
| Memory Throughput (%) | 25.24 % | 24.88 % | -0.36 pp | 24.79 % | -0.45 pp | 38.29 % | +13.05 pp | 24.69 % | -0.55 pp | 25.26 % | +0.02 pp |
| DRAM Throughput (%) | 15.24 % | 15.29 % | +0.05 pp | 15.34 % | +0.10 pp | 10.48 % | -4.76 pp | 15.30 % | +0.06 pp | 15.14 % | -0.10 pp |
| Achieved Occupancy (%) | 59.42 % | 41.59 % | -17.83 pp | 41.32 % | -18.10 pp | 79.50 % | +20.08 pp | 41.98 % | -17.44 pp | 60.31 % | +0.89 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 33.41 | +16.71 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 30.03 % | 57.04 % | +27.01 pp | 57.07 % | +27.04 pp | 23.22 % | -6.81 pp | 57.36 % | +27.33 pp | 29.59 % | -0.44 pp |
| IPC Active | 2.79 inst/cycle | 1.72 inst/cycle | -1.07 | 1.70 inst/cycle | -1.09 | 3.05 inst/cycle | +0.26 | 1.71 inst/cycle | -1.08 | 2.81 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0079× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_and_increase_num_warps_for_latency_hiding | 1.000× | 1.008× | ✓ |  |
| B | increase_num_warps_for_l1tex_latency_hiding | 1.001× | 1.009× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding_without_register_pressure | 1.007× | 1.005× | ✓ |  |
| D | revert_to_xblock128_with_reduced_num_warps_to_restore_register_budget | 1.001× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | revert_to_xblock1... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24350 ns | 24160 ns | -190 ns (-0.8%) | 24130 ns | -220 ns (-0.9%) | 24220 ns | -130 ns (-0.5%) | 24130 ns | -220 ns (-0.9%) |
| SM Throughput (%) | 62.92 % | 38.42 % | -24.50 pp | 38.51 % | -24.41 pp | 38.49 % | -24.43 pp | 38.47 % | -24.45 pp |
| Memory Throughput (%) | 25.24 % | 24.85 % | -0.39 pp | 24.90 % | -0.34 pp | 24.91 % | -0.33 pp | 24.88 % | -0.36 pp |
| DRAM Throughput (%) | 15.24 % | 15.37 % | +0.13 pp | 15.36 % | +0.12 pp | 15.32 % | +0.08 pp | 15.38 % | +0.14 pp |
| Achieved Occupancy (%) | 59.42 % | 41.69 % | -17.73 pp | 42.24 % | -17.18 pp | 41.92 % | -17.50 pp | 41.64 % | -17.78 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 30.03 % | 57.38 % | +27.35 pp | 57.47 % | +27.44 pp | 57.43 % | +27.40 pp | 56.97 % | +26.94 pp |
| IPC Active | 2.79 inst/cycle | 1.71 inst/cycle | -1.08 | 1.70 inst/cycle | -1.09 | 1.70 inst/cycle | -1.09 | 1.71 inst/cycle | -1.08 |

---

## Iteration 4

**Best so far**: 1.0079× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | apply_evict_first_on_streaming_tensor_to_reduce_l1tex_stall | 0.999× | 1.008× | ✓ |  |
| B | increase_block_size_to_256_with_8_warps_for_l1tex_latency_hiding | 1.009× | 1.008× | ✓ |  |
| C | apply_evict_first_to_streaming_tensor_load_to_reduce_l1tex_stalls | 1.001× | 1.008× | ✓ |  |
| D | apply_evict_first_to_streaming_main_tensor_load | 0.998× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | apply_evict_first... | Δ | increase_block_si... | Δ | apply_evict_first... | Δ | apply_evict_first... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24350 ns | 24160 ns | -190 ns (-0.8%) | 24160 ns | -190 ns (-0.8%) | 24160 ns | -190 ns (-0.8%) | 24160 ns | -190 ns (-0.8%) |
| SM Throughput (%) | 62.92 % | 42.21 % | -20.71 pp | 38.34 % | -24.58 pp | 42.22 % | -20.70 pp | 42.16 % | -20.76 pp |
| Memory Throughput (%) | 25.24 % | 24.96 % | -0.28 pp | 24.81 % | -0.43 pp | 24.99 % | -0.25 pp | 24.91 % | -0.33 pp |
| DRAM Throughput (%) | 15.24 % | 15.31 % | +0.07 pp | 15.35 % | +0.11 pp | 15.30 % | +0.06 pp | 15.32 % | +0.08 pp |
| Achieved Occupancy (%) | 59.42 % | 42.42 % | -17.00 pp | 41.71 % | -17.71 pp | 42.27 % | -17.15 pp | 42.20 % | -17.22 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 30.03 % | 53.06 % | +23.03 pp | 57.55 % | +27.52 pp | 52.87 % | +22.84 pp | 53.43 % | +23.40 pp |
| IPC Active | 2.79 inst/cycle | 1.88 inst/cycle | -0.91 | 1.72 inst/cycle | -1.07 | 1.88 inst/cycle | -0.91 | 1.88 inst/cycle | -0.91 |

---

## Cumulative Progress

```
Baseline:  24350 ns  (1.000×)
Iter 2     24160 ns  (1.008×)  — fix_broken_autotune_grid_and_fuse_rsqrt_instruction

Final best: 24160 ns  (1.0079×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -21,11 +22,7 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
@@ -34,8 +31,6 @@

     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
