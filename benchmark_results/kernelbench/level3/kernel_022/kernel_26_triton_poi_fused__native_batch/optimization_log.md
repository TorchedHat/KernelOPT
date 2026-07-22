# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7170 ns — **Best latency**: 7040 ns (**1.0185×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_2d_strided_grid_to_1d_contiguous_grid_for_coalesced_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_26_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7350,1,1), Waves/SM: 3.48
- **Registers/thread**: 18
- **Duration**: 7170 ns
- **SM Throughput**: 44.19 % | **Memory**: 16.19 % | **DRAM**: 10.73 %
- **Achieved Occupancy**: 58.98 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 34.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_reduce_partial_wave_and_l1tex_stalls | 0.952× | 0.789× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 1.019× | 0.996× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_better_occupancy | 0.996× | 0.475× | ✓ |  |
| D | increase_xblock_eliminate_partial_wave_and_add_autotune | 0.860× | 0.784× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7170 ns | 9090 ns | +1920 ns (+26.8%) | 7200 ns | +30 ns (+0.4%) | 15100 ns | +7930 ns (+110.6%) | 9150 ns | +1980 ns (+27.6%) |
| SM Throughput (%) | 44.19 % | 64.41 % | +20.22 pp | 44.79 % | +0.60 pp | 73.92 % | +29.73 pp | 64.61 % | +20.42 pp |
| Memory Throughput (%) | 16.19 % | 17.64 % | +1.45 pp | 16.53 % | +0.34 pp | 16.63 % | +0.44 pp | 17.64 % | +1.45 pp |
| DRAM Throughput (%) | 10.73 % | 8.47 % | -2.26 pp | 10.73 % | +0.00 pp | 5.10 % | -5.63 pp | 8.42 % | -2.31 pp |
| Achieved Occupancy (%) | 58.98 % | 80.94 % | +21.96 pp | 58.64 % | -0.34 pp | 84.05 % | +25.07 pp | 80.95 % | +21.97 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 6.96 | +3.48 | 3.48 | +0.00 |
| No Eligible (%) | 34.24 % | 17.56 % | -16.68 pp | 35.32 % | +1.08 pp | 14.82 % | -19.42 pp | 17.61 % | -16.63 pp |
| IPC Active | 2.48 inst/cycle | 3.28 inst/cycle | +0.80 | 2.54 inst/cycle | +0.06 | 3.40 inst/cycle | +0.92 | 3.26 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_to_reduce_partial_wave_and_l1tex_stalls` (0.789×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024, 2048) caused registers per thread to jump from 18 to 32, nearly doubling the total executed instruction count (2.56M → 4.65M) and increasing kernel duration from 7.17 µs to 9.09 µs despite improving achieved occupancy (59% → 81%). The register pressure increase from larger tile sizes forced the compiler to spill or use more registers, negating the occupancy and partial-wave benefits. When the original kernel already has low register usage (≤20 registers/thread), increasing XBLOCK via autotune can cause the compiler to allocate significantly more registers per thread, increasing instruction count and total runtime even if occupancy improves.

> **Applicable when**: Kernel has high partial-wave overhead (>20%), low achieved occupancy (<65%), and L1TEX stall rate >35% of cycles, with low register usage (<20 registers/thread) that leaves headroom for larger tiles.

> **Avoid when**: The kernel already has low register count (≤20 registers/thread) and the autotune configs include very large XBLOCK values (≥1024) that cause the compiler to allocate significantly more registers, doubling instruction count. Also avoid when the kernel is already compute-bound at the original tile size, as larger tiles increase instruction count without proportional throughput gains.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_software_pipelining_via_num_stages_at_fixed_xblock | 1.018× | 0.974× | ✓ |  |
| B | replace_integer_reciprocal_division_with_float_reciprocal_to_enable_fma_fusion | 1.030× | 0.970× | ✓ | **✓ best** |
| C | replace_int_div_sqrt_chain_with_fused_rsqrt_to_reduce_nonfused_fp32_instructions | 0.979× | 0.970× | ✓ |  |
| D | replace_integer_reciprocal_sqrt_with_tl_rsqrt_to_enable_fma_fusion | 1.006× | 0.798× | ✓ |  |
| E | add_triton_autotune | 0.995× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_software_pipe... | Δ | replace_integer_r... | Δ | replace_int_div_s... | Δ | replace_integer_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7170 ns | 7360 ns | +190 ns (+2.6%) | 7390 ns | +220 ns (+3.1%) | 7390 ns | +220 ns (+3.1%) | 8990 ns | +1820 ns (+25.4%) | 7200 ns | +30 ns (+0.4%) |
| SM Throughput (%) | 44.19 % | 44.62 % | +0.43 pp | 36.74 % | -7.45 pp | 37.03 % | -7.16 pp | 59.71 % | +15.52 pp | 44.68 % | +0.49 pp |
| Memory Throughput (%) | 16.19 % | 16.35 % | +0.16 pp | 23.89 % | +7.70 pp | 24.06 % | +7.87 pp | 29.76 % | +13.57 pp | 16.44 % | +0.25 pp |
| DRAM Throughput (%) | 10.73 % | 10.48 % | -0.25 pp | 10.47 % | -0.26 pp | 10.45 % | -0.28 pp | 8.56 % | -2.17 pp | 10.71 % | -0.02 pp |
| Achieved Occupancy (%) | 58.98 % | 59.85 % | +0.87 pp | 43.48 % | -15.50 pp | 45.12 % | -13.86 pp | 77.24 % | +18.26 pp | 58.60 % | -0.38 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 | 6.96 | +3.48 | 3.48 | +0.00 |
| No Eligible (%) | 34.24 % | 34.01 % | -0.23 pp | 39.87 % | +5.63 pp | 39.20 % | +4.96 pp | 24.45 % | -9.79 pp | 33.67 % | -0.57 pp |
| IPC Active | 2.48 inst/cycle | 2.57 inst/cycle | +0.09 | 2.40 inst/cycle | -0.08 | 2.37 inst/cycle | -0.11 | 3.00 inst/cycle | +0.52 | 2.53 inst/cycle | +0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_eviction_hints_to_streaming_tensor_to_reduce_l1tex_stall_and_l2_pressure | 1.014× | 0.978× | ✓ |  |
| B | cap_register_usage_with_maxnreg_to_recover_occupancy | 1.041× | 1.023× | ✓ | **✓ best** |
| C | set_evict_first_on_streaming_tensor_load_to_prioritize_broadcast_channel_params_in_l1 | 0.979× | 0.974× | ✓ |  |
| D | revert_to_xblock128_and_apply_isolated_fma_fusion | 1.020× | 1.023× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_eviction_hint... | Δ | cap_register_usag... | Δ | set_evict_first_o... | Δ | revert_to_xblock1... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7170 ns | 7330 ns | +160 ns (+2.2%) | 7010 ns | -160 ns (-2.2%) | 7360 ns | +190 ns (+2.6%) | 7010 ns | -160 ns (-2.2%) |
| SM Throughput (%) | 44.19 % | 48.67 % | +4.48 pp | 27.55 % | -16.64 pp | 38.76 % | -5.43 pp | 26.92 % | -17.27 pp |
| Memory Throughput (%) | 16.19 % | 16.10 % | -0.09 pp | 17.07 % | +0.88 pp | 23.98 % | +7.79 pp | 16.40 % | +0.21 pp |
| DRAM Throughput (%) | 10.73 % | 10.52 % | -0.21 pp | 11.01 % | +0.28 pp | 10.48 % | -0.25 pp | 11.05 % | +0.32 pp |
| Achieved Occupancy (%) | 58.98 % | 61.86 % | +2.88 pp | 48.23 % | -10.75 pp | 45.24 % | -13.74 pp | 49.70 % | -9.28 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 34.24 % | 28.49 % | -5.75 pp | 58.13 % | +23.89 pp | 35.77 % | +1.53 pp | 57.39 % | +23.15 pp |
| IPC Active | 2.48 inst/cycle | 2.78 inst/cycle | +0.30 | 1.69 inst/cycle | -0.79 | 2.51 inst/cycle | +0.03 | 1.62 inst/cycle | -0.86 |

---

## Iteration 4

**Best so far**: 1.0228× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_to_2d_grid_to_eliminate_modulo_and_improve_channel_load_coalescing | 0.205× | 0.076× | ✓ |  |
| C | fuse_sqrt_reciprocal_chain_into_rsqrt_to_eliminate_nonfused_fp32_instructions | 1.015× | 1.018× | ✓ |  |
| D | fix_broken_autotune_grid_lambda_and_tune_num_warps_at_fixed_xblock | 1.041× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_to_2d... | Δ | fuse_sqrt_recipro... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7170 ns | 93890 ns | +86720 ns (+1209.5%) | 7040 ns | -130 ns (-1.8%) | 7170 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 44.19 % | 2.25 % | -41.94 pp | 29.34 % | -14.85 pp | 11.29 % | -32.90 pp |
| Memory Throughput (%) | 16.19 % | 25.09 % | +8.90 pp | 16.60 % | +0.41 pp | 15.32 % | -0.87 pp |
| DRAM Throughput (%) | 10.73 % | 0.08 % | -10.65 pp | 10.97 % | +0.24 pp | 10.75 % | +0.02 pp |
| Achieved Occupancy (%) | 58.98 % | 80.21 % | +21.23 pp | 50.24 % | -8.74 pp | 18.21 % | -40.77 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 1.74 | -1.74 |
| No Eligible (%) | 34.24 % | 97.77 % | +63.53 pp | 53.55 % | +19.31 pp | 74.54 % | +40.30 pp |
| IPC Active | 2.48 inst/cycle | 0.09 inst/cycle | -2.39 | 1.75 inst/cycle | -0.73 | 0.88 inst/cycle | -1.60 |


### Key Learnings

#### ❌ Plan A: `restructure_to_2d_grid_to_eliminate_modulo_and_improve_channel_load_coalescing` (0.076×)

Restructuring a flat 1D grid to a 2D grid (channel × spatial) to eliminate an integer modulo operation caused a 13x slowdown: kernel duration jumped from 7.01 µs to 93.89 µs, Warp Cycles Per Issued Instruction exploded from 18.82 to 603.42, and eligible warps per scheduler collapsed from 1.15 to 0.05. The root cause is that the 2D grid changed the in-place tensor access pattern from contiguous (stride-1 sequential) to heavily strided (stride=C=480 floats=1920 bytes between consecutive threads), triggering the UncoalescedGlobalAccess rule with 82% excessive sectors (1.65M of 2.0M total). Although L2 hit rate improved (55%→92%) due to the smaller working set per block, the uncoalesced access pattern caused catastrophic L1TEX stall amplification (CPIStall cycles per warp: 10.1→275.3), completely overwhelming any benefit from eliminating the modulo.

> **Applicable when**: Never apply this specific pattern when the tensor being read/written in-place has a channel-last or interleaved layout where the channel stride is large (>1 element), as the 2D grid will make the primary tensor access uncoalesced.

> **Avoid when**: The primary in-place tensor has a non-unit stride between consecutive threads in the spatial dimension (i.e., channel-last or NHWC layout with C>1). In such layouts, a 2D grid with spatial indexing as xindex*C+pid_c creates stride-C access patterns that are maximally uncoalesced, causing Warp Cycles Per Issued Instruction to explode by 30x+ and eligible warps per scheduler to collapse near zero. The modulo cost is negligible compared to the uncoalesced memory penalty.

---

## Iteration 5

**Best so far**: 1.0228× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.041× | 1.010× | ✓ |  |
| B | force_num_warps4_at_fixed_xblock128_to_hide_l1tex_latency | 1.017× | 1.023× | ✓ |  |
| C | revert_2d_strided_grid_to_1d_contiguous_grid_for_coalesced_access | 1.045× | 1.018× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | force_num_warps4_... | Δ | revert_2d_strided... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7170 ns | 7100 ns | -70 ns (-1.0%) | 7010 ns | -160 ns (-2.2%) | 7040 ns | -130 ns (-1.8%) |
| SM Throughput (%) | 44.19 % | 29.99 % | -14.20 pp | 30.70 % | -13.49 pp | 28.66 % | -15.53 pp |
| Memory Throughput (%) | 16.19 % | 16.95 % | +0.76 pp | 17.10 % | +0.91 pp | 17.19 % | +1.00 pp |
| DRAM Throughput (%) | 10.73 % | 10.90 % | +0.17 pp | 11.01 % | +0.28 pp | 10.97 % | +0.24 pp |
| Achieved Occupancy (%) | 58.98 % | 47.31 % | -11.67 pp | 47.21 % | -11.77 pp | 48.34 % | -10.64 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 34.24 % | 53.90 % | +19.66 pp | 53.12 % | +18.88 pp | 56.03 % | +21.79 pp |
| IPC Active | 2.48 inst/cycle | 1.83 inst/cycle | -0.65 | 1.86 inst/cycle | -0.62 | 1.75 inst/cycle | -0.73 |

---

## Cumulative Progress

```
Baseline:  7170 ns  (1.000×)
Iter 3     7010 ns  (1.023×)  — cap_register_usage_with_maxnreg_to_recover_occupancy

Final best: 7040 ns  (1.0185×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
+], key=['xnumel'], restore_value=['in_out_ptr0'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 940800
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x2 = xindex
-    x0 = (xindex % 480)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
+    xmask   = xindex < xnumel
+    x0 = xindex % 480          # channel index via modulo
+    tmp1  = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')   # mean[c]
+    tmp3  = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')   # var[c]
+    tmp12 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')   # weight[c]
+    tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')   # bias[c]
+    tmp0  = tl.load(in_out_ptr0 + xindex, xmask)   # COALESCED: stride-1 access
+    tmp2  = tmp0 - tmp1
+    tmp4  = tl.full([1], 1e-05, tl.float32)
+    tmp5  = tmp3 + tmp4
+    tmp15 = tmp2 * tl.math.rsqrt(tmp5) * tmp12 + tmp14
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
+    tl.store(in_out_ptr0 + xindex, tmp19, xmask)   # COALESCED: stride-1 store
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +42,7 @@

     arg_4 = inputs[4]
     xnumel = 940800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 940800, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 940800)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
