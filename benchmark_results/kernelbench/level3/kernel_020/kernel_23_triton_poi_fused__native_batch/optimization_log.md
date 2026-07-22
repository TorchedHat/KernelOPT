# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7870 ns — **Best latency**: 7650 ns (**1.0288×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_xblock_to_64_with_2_warps_to_increase_concurrent_blocks_and_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_23_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8438,1,1), Waves/SM: 4.00
- **Registers/thread**: 18
- **Duration**: 7870 ns
- **SM Throughput**: 47.35 % | **Memory**: 17.52 % | **DRAM**: 11.25 %
- **Achieved Occupancy**: 61.06 %
- **IPC Active**: 2.64 inst/cycle | **No Eligible**: 32.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.808× | 0.764× | ✓ |  |
| B | increase_xblock_and_autotune_for_occupancy_and_tail_elimination | 0.822× | 0.759× | ✓ |  |
| C | vectorized_loads_larger_xblock_autotune | 0.816× | 0.741× | ✓ |  |
| D | autotune_xblock_eliminate_partial_wave | 0.808× | 0.759× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7870 ns | 10300 ns | +2430 ns (+30.9%) | 10370 ns | +2500 ns (+31.8%) | 10620 ns | +2750 ns (+34.9%) | 10370 ns | +2500 ns (+31.8%) |
| SM Throughput (%) | 47.35 % | 66.42 % | +19.07 pp | 65.99 % | +18.64 pp | 65.84 % | +18.49 pp | 65.92 % | +18.57 pp |
| Memory Throughput (%) | 17.52 % | 18.26 % | +0.74 pp | 18.05 % | +0.53 pp | 18.42 % | +0.90 pp | 18.18 % | +0.66 pp |
| DRAM Throughput (%) | 11.25 % | 8.61 % | -2.64 pp | 8.55 % | -2.70 pp | 8.44 % | -2.81 pp | 8.55 % | -2.70 pp |
| Achieved Occupancy (%) | 61.06 % | 82.07 % | +21.01 pp | 82.35 % | +21.29 pp | 82.47 % | +21.41 pp | 82.17 % | +21.11 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 4.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 |
| No Eligible (%) | 32.57 % | 17.34 % | -15.23 pp | 18.46 % | -14.11 pp | 17.53 % | -15.04 pp | 17.33 % | -15.24 pp |
| IPC Active | 2.64 inst/cycle | 3.26 inst/cycle | +0.62 | 3.24 inst/cycle | +0.60 | 3.26 inst/cycle | +0.62 | 3.25 inst/cycle | +0.61 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_non_fused_fp32_sqrt_reciprocal_chain_with_rsqrt | 0.996× | 1.021× | ✓ |  |
| B | replace_sqrt_reciprocal_with_rsqrt_to_reduce_instruction_count_and_register_pressure | 1.020× | 1.029× | ✓ | **✓ best** |
| C | reduce_nonfused_fp32_instructions_via_fma | 0.997× | 1.017× | ✓ |  |
| D | fuse_batchnorm_arithmetic_with_rsqrt_to_eliminate_nonfused_fp32_instructions | 0.943× | 0.925× | ✓ |  |
| E | add_triton_autotune | 1.000× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_non_fused... | Δ | replace_sqrt_reci... | Δ | reduce_nonfused_f... | Δ | fuse_batchnorm_ar... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7870 ns | 7710 ns | -160 ns (-2.0%) | 7650 ns | -220 ns (-2.8%) | 7740 ns | -130 ns (-1.7%) | 8510 ns | +640 ns (+8.1%) | 7940 ns | +70 ns (+0.9%) |
| SM Throughput (%) | 47.35 % | 29.58 % | -17.77 pp | 29.60 % | -17.75 pp | 29.61 % | -17.74 pp | 38.59 % | -8.76 pp | 46.89 % | -0.46 pp |
| Memory Throughput (%) | 17.52 % | 18.13 % | +0.61 pp | 18.12 % | +0.60 pp | 18.14 % | +0.62 pp | 24.72 % | +7.20 pp | 17.36 % | -0.16 pp |
| DRAM Throughput (%) | 11.25 % | 11.47 % | +0.22 pp | 11.56 % | +0.31 pp | 11.47 % | +0.22 pp | 10.40 % | -0.85 pp | 11.13 % | -0.12 pp |
| Achieved Occupancy (%) | 61.06 % | 49.13 % | -11.93 pp | 49.40 % | -11.66 pp | 46.68 % | -14.38 pp | 43.55 % | -17.51 pp | 61.83 % | +0.77 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 4.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 |
| No Eligible (%) | 32.57 % | 56.84 % | +24.27 pp | 56.49 % | +23.92 pp | 56.45 % | +23.88 pp | 41.32 % | +8.75 pp | 34.48 % | +1.91 pp |
| IPC Active | 2.64 inst/cycle | 1.66 inst/cycle | -0.98 | 1.65 inst/cycle | -0.99 | 1.74 inst/cycle | -0.90 | 2.31 inst/cycle | -0.33 | 2.60 inst/cycle | -0.04 |

---

## Iteration 3

**Best so far**: 1.0288× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | moderate_xblock_increase_with_correct_grid_lambda_to_hide_l1tex_latency | 0.992× | 1.017× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_scoreboard_latency | 0.998× | 1.021× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_scoreboard_stall_latency | 0.997× | 1.021× | ✓ |  |
| D | fix_autotune_grid_lambda_to_enable_correct_tile_sizing | 0.945× | 0.946× | ✓ |  |


### Metric Comparison

| Metric | Baseline | moderate_xblock_i... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7870 ns | 7740 ns | -130 ns (-1.7%) | 7710 ns | -160 ns (-2.0%) | 7710 ns | -160 ns (-2.0%) | 8320 ns | +450 ns (+5.7%) |
| SM Throughput (%) | 47.35 % | 30.04 % | -17.31 pp | 29.39 % | -17.96 pp | 29.27 % | -18.08 pp | 38.54 % | -8.81 pp |
| Memory Throughput (%) | 17.52 % | 18.09 % | +0.57 pp | 18.01 % | +0.49 pp | 17.92 % | +0.40 pp | 24.69 % | +7.17 pp |
| DRAM Throughput (%) | 11.25 % | 11.41 % | +0.16 pp | 11.46 % | +0.21 pp | 11.47 % | +0.22 pp | 10.63 % | -0.62 pp |
| Achieved Occupancy (%) | 61.06 % | 48.10 % | -12.96 pp | 46.01 % | -15.05 pp | 46.32 % | -14.74 pp | 43.87 % | -17.19 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 4.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 |
| No Eligible (%) | 32.57 % | 56.78 % | +24.21 pp | 56.39 % | +23.82 pp | 56.86 % | +24.29 pp | 41.16 % | +8.59 pp |
| IPC Active | 2.64 inst/cycle | 1.75 inst/cycle | -0.89 | 1.68 inst/cycle | -0.96 | 1.73 inst/cycle | -0.91 | 2.32 inst/cycle | -0.32 |

---

## Iteration 4

**Best so far**: 1.0288× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_main_tensor_eviction_to_evict_first_to_prioritize_broadcast_vector_l1_residency | 1.021× | 1.029× | ✓ | **✓ best** |
| B | evict_first_on_streaming_load_to_improve_broadcast_cache_reuse | 0.998× | 1.029× | ✓ |  |
| C | increase_warps_per_block_via_xblock256_to_hide_l1tex_scoreboard_latency | 1.017× | 1.021× | ✓ |  |
| D | software_pipeline_l1tex_stalls_via_num_stages | 1.015× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_main_tenso... | Δ | evict_first_on_st... | Δ | increase_warps_pe... | Δ | software_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7870 ns | 7650 ns | -220 ns (-2.8%) | 7650 ns | -220 ns (-2.8%) | 7710 ns | -160 ns (-2.0%) | 7780 ns | -90 ns (-1.1%) |
| SM Throughput (%) | 47.35 % | 31.81 % | -15.54 pp | 31.43 % | -15.92 pp | 30.09 % | -17.26 pp | 17.65 % | -29.70 pp |
| Memory Throughput (%) | 17.52 % | 18.07 % | +0.55 pp | 17.57 % | +0.05 pp | 18.11 % | +0.59 pp | 16.65 % | -0.87 pp |
| DRAM Throughput (%) | 11.25 % | 11.56 % | +0.31 pp | 11.56 % | +0.31 pp | 11.52 % | +0.27 pp | 11.40 % | +0.15 pp |
| Achieved Occupancy (%) | 61.06 % | 48.72 % | -12.34 pp | 48.74 % | -12.32 pp | 49.93 % | -11.13 pp | 36.32 % | -24.74 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 22.00 register/thread | +4 |
| Waves/SM | 4.00 | 4.00 | +0.00 | 4.00 | +0.00 | 4.00 | +0.00 | 2.00 | -2.00 |
| No Eligible (%) | 32.57 % | 53.68 % | +21.11 pp | 52.35 % | +19.78 pp | 55.73 % | +23.16 pp | 68.53 % | +35.96 pp |
| IPC Active | 2.64 inst/cycle | 1.81 inst/cycle | -0.83 | 1.86 inst/cycle | -0.78 | 1.68 inst/cycle | -0.96 | 1.24 inst/cycle | -1.40 |

---

## Iteration 5

**Best so far**: 1.0288× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_channel_broadcast_params_into_shared_memory | 1.012× | 1.025× | ✓ |  |
| B | reduce_xblock_to_64_with_2_warps_to_increase_concurrent_blocks_and_hide_l1tex_latency | 0.995× | 1.029× | ✓ |  |
| C | eliminate_non_power_of_two_modulo_via_2d_grid_indexing | 0.498× | 0.421× | ✓ |  |
| D | cap_registers_with_maxnreg_to_increase_warp_density_and_hide_l1tex_latency | 0.991× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_channel_b... | Δ | reduce_xblock_to_... | Δ | eliminate_non_pow... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7870 ns | 7680 ns | -190 ns (-2.4%) | 7650 ns | -220 ns (-2.8%) | 18690 ns | +10820 ns (+137.5%) | 7680 ns | -190 ns (-2.4%) |
| SM Throughput (%) | 47.35 % | 26.82 % | -20.53 pp | 32.36 % | -14.99 pp | 65.12 % | +17.77 pp | 30.17 % | -17.18 pp |
| Memory Throughput (%) | 17.52 % | 17.74 % | +0.22 pp | 18.11 % | +0.59 pp | 18.24 % | +0.72 pp | 18.08 % | +0.56 pp |
| DRAM Throughput (%) | 11.25 % | 11.53 % | +0.28 pp | 11.54 % | +0.29 pp | 0.28 % | -10.97 pp | 11.52 % | +0.27 pp |
| Achieved Occupancy (%) | 61.06 % | 48.63 % | -12.43 pp | 47.56 % | -13.50 pp | 37.99 % | -23.07 pp | 47.73 % | -13.33 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 72.00 register/thread | +54 | 18.00 register/thread | +0 |
| Waves/SM | 4.00 | 4.00 | +0.00 | 4.00 | +0.00 | 9.13 | +5.13 | 4.00 | +0.00 |
| No Eligible (%) | 32.57 % | 60.63 % | +28.06 pp | 52.85 % | +20.28 pp | 27.44 % | -5.13 pp | 55.84 % | +23.27 pp |
| IPC Active | 2.64 inst/cycle | 1.52 inst/cycle | -1.12 | 1.89 inst/cycle | -0.75 | 2.90 inst/cycle | +0.26 | 1.77 inst/cycle | -0.87 |


### Key Learnings

#### ❌ Plan C: `eliminate_non_power_of_two_modulo_via_2d_grid_indexing` (0.421×)

Restructuring a 1D flat pointwise kernel into a 2D grid (channel × spatial) to eliminate a non-power-of-2 modulo operation caused registers per thread to explode from 18 to 72, reducing theoretical occupancy from 100% to 43.75% and achieved occupancy from 48.7% to 38.0%. The 2D indexing introduced 2D tensor shapes ([XBLOCK_N, XBLOCK_C]) that the Triton compiler could not efficiently register-allocate, increasing executed instructions by 5× (1.9M → 9.7M) and kernel duration by 2.4× (7.65 µs → 18.69 µs). The L1/TEX hit rate also degraded (79.4% → 51.5%) and effective memory throughput collapsed from 566 GB/s to 14 GB/s, indicating the 2D tiling destroyed the cache access pattern that the original 1D kernel exploited.

> **Applicable when**: Never: this specific 2D grid restructuring approach is harmful for short pointwise kernels where the Triton compiler allocates 2D intermediate tensors that balloon register usage.

> **Avoid when**: Do not restructure a 1D pointwise Triton kernel into a 2D grid (channel × spatial) to eliminate a non-power-of-2 modulo when: (1) the kernel has low register pressure at baseline (≤20 regs/thread); (2) the 2D tiling introduces 2D shaped intermediate tensors ([XBLOCK_N, XBLOCK_C]) that the compiler must hold in registers simultaneously — this can multiply register usage by 4× or more; (3) the original 1D kernel already achieves high L1/TEX hit rate (>75%) via sequential access patterns that the 2D tiling would disrupt; (4) the kernel duration is short (<10 µs) and the grid is already producing multiple full waves. The modulo cost (a few integer division cycles) is far smaller than the cost of 4× register pressure increase and 5× instruction count explosion.

---

## Cumulative Progress

```
Baseline:  7870 ns  (1.000×)
Iter 2     7650 ns  (1.029×)  — replace_sqrt_reciprocal_with_rsqrt_to_reduce_instruction_count_and_register_pressure

Final best: 7650 ns  (1.0288×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -13,7 +14,7 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 480)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
@@ -21,11 +22,7 @@

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
@@ -33,9 +30,7 @@

     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
