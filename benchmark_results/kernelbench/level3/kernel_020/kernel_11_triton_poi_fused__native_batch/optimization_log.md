# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24380 ns — **Best latency**: 24160 ns (**1.0091×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_evict_first_on_main_streaming_load_and_store_to_free_l1_for_channel_params`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (35280,1,1), Waves/SM: 16.70
- **Registers/thread**: 18
- **Duration**: 24380 ns
- **SM Throughput**: 63.76 % | **Memory**: 25.50 % | **DRAM**: 15.14 %
- **Achieved Occupancy**: 60.28 %
- **IPC Active**: 2.80 inst/cycle | **No Eligible**: 29.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_improve_occupancy_and_hide_l1tex_latency | 0.999× | 0.206× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 0.926× | 0.942× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock_autotune | 0.914× | 0.206× | ✓ |  |
| D | autotune_xblock_num_warps_for_occupancy_and_latency_hiding | 1.007× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_i... | Δ | vectorized_loads_... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24380 ns | 118300 ns | +93920 ns (+385.2%) | 25890 ns | +1510 ns (+6.2%) | 118340 ns | +93960 ns (+385.4%) | 24380 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 63.76 % | 85.27 % | +21.51 pp | 50.46 % | -13.30 pp | 85.29 % | +21.53 pp | 63.37 % | -0.39 pp |
| Memory Throughput (%) | 25.50 % | 18.12 % | -7.38 pp | 36.02 % | +10.52 pp | 18.15 % | -7.35 pp | 25.35 % | -0.15 pp |
| DRAM Throughput (%) | 15.14 % | 3.12 % | -12.02 pp | 14.27 % | -0.87 pp | 3.12 % | -12.02 pp | 15.13 % | -0.01 pp |
| Achieved Occupancy (%) | 60.28 % | 55.83 % | -4.45 pp | 39.41 % | -20.87 pp | 55.90 % | -4.38 pp | 60.67 % | +0.39 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 32.00 register/thread | +14 | 48.00 register/thread | +30 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 53.45 | +36.75 | 16.70 | +0.00 | 53.45 | +36.75 | 16.70 | +0.00 |
| No Eligible (%) | 29.53 % | 13.14 % | -16.39 pp | 43.05 % | +13.52 pp | 13.10 % | -16.43 pp | 29.44 % | -0.09 pp |
| IPC Active | 2.80 inst/cycle | 3.47 inst/cycle | +0.67 | 2.28 inst/cycle | -0.52 | 3.47 inst/cycle | +0.67 | 2.82 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_int_division_with_fp32_rsqrt_and_add_evict_first_on_streaming_load | 1.007× | 1.001× | ✓ |  |
| B | cap_register_count_with_maxnreg_to_increase_warp_residency_and_hide_l1tex_latency | 1.013× | 1.008× | ✓ |  |
| C | reduce_register_pressure_by_removing_autotune_and_fixing_small_xblock | 1.000× | 0.992× | ✓ |  |
| D | revert_register_pressure_and_fuse_fp_instructions_via_rsqrt | 1.018× | 1.009× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_int_divis... | Δ | cap_register_coun... | Δ | reduce_register_p... | Δ | revert_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24380 ns | 24350 ns | -30 ns (-0.1%) | 24190 ns | -190 ns (-0.8%) | 24580 ns | +200 ns (+0.8%) | 24160 ns | -220 ns (-0.9%) |
| SM Throughput (%) | 63.76 % | 42.28 % | -21.48 pp | 37.78 % | -25.98 pp | 63.12 % | -0.64 pp | 38.32 % | -25.44 pp |
| Memory Throughput (%) | 25.50 % | 25.03 % | -0.47 pp | 24.99 % | -0.51 pp | 25.31 % | -0.19 pp | 24.85 % | -0.65 pp |
| DRAM Throughput (%) | 15.14 % | 15.16 % | +0.02 pp | 15.28 % | +0.14 pp | 15.04 % | -0.10 pp | 15.30 % | +0.16 pp |
| Achieved Occupancy (%) | 60.28 % | 42.88 % | -17.40 pp | 42.04 % | -18.24 pp | 60.71 % | +0.43 pp | 42.10 % | -18.18 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 29.53 % | 52.89 % | +23.36 pp | 57.89 % | +28.36 pp | 29.69 % | +0.16 pp | 57.13 % | +27.60 pp |
| IPC Active | 2.80 inst/cycle | 1.88 inst/cycle | -0.92 | 1.68 inst/cycle | -1.12 | 2.82 inst/cycle | +0.02 | 1.71 inst/cycle | -1.09 |

---

## Iteration 3

**Best so far**: 1.0091× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_scoreboard_stalls | 1.009× | 1.008× | ✓ |  |
| B | fuse_fp32_instructions_with_fma_and_evict_first_on_streaming_load | 1.016× | 1.005× | ✓ |  |
| C | fix_autotune_grid_and_tune_num_warps_for_l1tex_latency_hiding | 1.018× | 0.997× | ✓ |  |
| D | fuse_non_fused_fp32_instructions_via_libdevice_rsqrt | 1.009× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fuse_fp32_instruc... | Δ | fix_autotune_grid... | Δ | fuse_non_fused_fp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24380 ns | 24190 ns | -190 ns (-0.8%) | 24260 ns | -120 ns (-0.5%) | 24450 ns | +70 ns (+0.3%) | 24260 ns | -120 ns (-0.5%) |
| SM Throughput (%) | 63.76 % | 38.31 % | -25.45 pp | 40.76 % | -23.00 pp | 24.21 % | -39.55 pp | 38.27 % | -25.49 pp |
| Memory Throughput (%) | 25.50 % | 24.80 % | -0.70 pp | 25.02 % | -0.48 pp | 22.81 % | -2.69 pp | 24.82 % | -0.68 pp |
| DRAM Throughput (%) | 15.14 % | 15.29 % | +0.15 pp | 15.22 % | +0.08 pp | 15.09 % | -0.05 pp | 15.25 % | +0.11 pp |
| Achieved Occupancy (%) | 60.28 % | 42.14 % | -18.14 pp | 42.80 % | -17.48 pp | 27.27 % | -33.01 pp | 42.29 % | -17.99 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 8.35 | -8.35 | 16.70 | +0.00 |
| No Eligible (%) | 29.53 % | 57.32 % | +27.79 pp | 54.95 % | +25.42 pp | 70.95 % | +41.42 pp | 56.96 % | +27.43 pp |
| IPC Active | 2.80 inst/cycle | 1.72 inst/cycle | -1.08 | 1.80 inst/cycle | -1.00 | 1.13 inst/cycle | -1.67 | 1.71 inst/cycle | -1.09 |

---

## Iteration 4

**Best so far**: 1.0091× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | collapse_arithmetic_chain_into_explicit_fma_to_reduce_non_fused_fp32_instructions | 1.018× | 1.004× | ✓ |  |
| B | change_store_eviction_policy_to_evict_last_for_l2_reuse_on_inplace_tensor | 1.006× | 1.008× | ✓ |  |
| C | add_evict_first_on_main_streaming_load_and_store_to_free_l1_for_channel_params | 1.017× | 1.009× | ✓ |  |
| D | fuse_fp32_multiply_add_chain_with_tl_math_fma_to_reduce_instruction_count | 1.017× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | collapse_arithmet... | Δ | change_store_evic... | Δ | add_evict_first_o... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24380 ns | 24290 ns | -90 ns (-0.4%) | 24190 ns | -190 ns (-0.8%) | 24160 ns | -220 ns (-0.9%) | 24260 ns | -120 ns (-0.5%) |
| SM Throughput (%) | 63.76 % | 38.29 % | -25.47 pp | 40.82 % | -22.94 pp | 41.17 % | -22.59 pp | 24.40 % | -39.36 pp |
| Memory Throughput (%) | 25.50 % | 24.70 % | -0.80 pp | 24.98 % | -0.52 pp | 24.81 % | -0.69 pp | 22.97 % | -2.53 pp |
| DRAM Throughput (%) | 15.14 % | 15.19 % | +0.05 pp | 15.22 % | +0.08 pp | 15.27 % | +0.13 pp | 15.22 % | +0.08 pp |
| Achieved Occupancy (%) | 60.28 % | 42.47 % | -17.81 pp | 42.60 % | -17.68 pp | 43.31 % | -16.97 pp | 27.31 % | -32.97 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 22.00 register/thread | +4 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 | 8.35 | -8.35 |
| No Eligible (%) | 29.53 % | 57.17 % | +27.64 pp | 54.96 % | +25.43 pp | 54.04 % | +24.51 pp | 71.37 % | +41.84 pp |
| IPC Active | 2.80 inst/cycle | 1.71 inst/cycle | -1.09 | 1.81 inst/cycle | -0.99 | 1.84 inst/cycle | -0.96 | 1.13 inst/cycle | -1.67 |

---

## Cumulative Progress

```
Baseline:  24380 ns  (1.000×)
Iter 2     24160 ns  (1.009×)  — revert_register_pressure_and_fuse_fp_instructions_via_rsqrt

Final best: 24160 ns  (1.0091×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -21,11 +22,7 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.rsqrt(tmp5)
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

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
