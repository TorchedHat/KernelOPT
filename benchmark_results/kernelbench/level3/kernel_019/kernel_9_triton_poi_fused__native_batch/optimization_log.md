# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7420 ns — **Best latency**: 7260 ns (**1.0220×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`enable_fma_fusion_and_evict_first_on_streaming_load`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7420 ns
- **SM Throughput**: 38.23 % | **Memory**: 16.65 % | **DRAM**: 11.06 %
- **Achieved Occupancy**: 57.16 %
- **IPC Active**: 2.25 inst/cycle | **No Eligible**: 42.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_latency | 0.969× | 0.916× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_latency_hiding | 0.810× | 0.775× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.836× | 0.775× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.853× | 0.758× | ✓ |  |
| E | add_triton_autotune | 1.018× | 0.991× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 8100 ns | +680 ns (+9.2%) | 9570 ns | +2150 ns (+29.0%) | 9570 ns | +2150 ns (+29.0%) | 9790 ns | +2370 ns (+31.9%) | 7490 ns | +70 ns (+0.9%) |
| SM Throughput (%) | 38.23 % | 55.82 % | +17.59 pp | 66.17 % | +27.94 pp | 66.56 % | +28.33 pp | 66.48 % | +28.25 pp | 45.86 % | +7.63 pp |
| Memory Throughput (%) | 16.65 % | 20.43 % | +3.78 pp | 17.97 % | +1.32 pp | 18.13 % | +1.48 pp | 18.04 % | +1.39 pp | 17.17 % | +0.52 pp |
| DRAM Throughput (%) | 11.06 % | 10.15 % | -0.91 pp | 8.58 % | -2.48 pp | 8.57 % | -2.49 pp | 8.38 % | -2.68 pp | 11.00 % | -0.06 pp |
| Achieved Occupancy (%) | 57.16 % | 71.42 % | +14.26 pp | 82.12 % | +24.96 pp | 81.39 % | +24.23 pp | 81.37 % | +24.21 pp | 60.16 % | +3.00 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.71 % | 19.15 % | -23.56 pp | 15.61 % | -27.10 pp | 15.50 % | -27.21 pp | 15.71 % | -27.00 pp | 33.45 % | -9.26 pp |
| IPC Active | 2.25 inst/cycle | 3.28 inst/cycle | +1.03 | 3.24 inst/cycle | +0.99 | 3.34 inst/cycle | +1.09 | 3.34 inst/cycle | +1.09 | 2.64 inst/cycle | +0.39 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_tile_size_selection | 0.998× | 0.974× | ✓ |  |
| B | replace_nonfused_fp32_arithmetic_with_fma_friendly_rsqrt_chain | 1.001× | 0.991× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_bloat | 0.975× | 0.954× | ✓ |  |
| D | fix_autotune_grid_to_use_lambda_meta_expression | 0.997× | 0.962× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | replace_nonfused_... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7620 ns | +200 ns (+2.7%) | 7490 ns | +70 ns (+0.9%) | 7780 ns | +360 ns (+4.9%) | 7710 ns | +290 ns (+3.9%) |
| SM Throughput (%) | 38.23 % | 45.59 % | +7.36 pp | 30.80 % | -7.43 pp | 37.99 % | -0.24 pp | 45.84 % | +7.61 pp |
| Memory Throughput (%) | 16.65 % | 17.05 % | +0.40 pp | 22.97 % | +6.32 pp | 24.18 % | +7.53 pp | 17.13 % | +0.48 pp |
| DRAM Throughput (%) | 11.06 % | 10.76 % | -0.30 pp | 10.96 % | -0.10 pp | 10.56 % | -0.50 pp | 10.68 % | -0.38 pp |
| Achieved Occupancy (%) | 57.16 % | 60.77 % | +3.61 pp | 41.34 % | -15.82 pp | 43.39 % | -13.77 pp | 60.81 % | +3.65 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.71 % | 33.17 % | -9.54 pp | 46.97 % | +4.26 pp | 40.57 % | -2.14 pp | 33.68 % | -9.03 pp |
| IPC Active | 2.25 inst/cycle | 2.63 inst/cycle | +0.38 | 2.10 inst/cycle | -0.15 | 2.38 inst/cycle | +0.13 | 2.58 inst/cycle | +0.33 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_broken_autotune_and_set_optimal_fixed_config_with_higher_num_warps | 1.015× | 1.022× | ✓ |  |
| B | replace_int_div_sqrt_chain_with_rsqrt_to_enable_fma_fusion | 1.018× | 1.026× | ✓ |  |
| C | replace_int_div_sqrt_chain_with_rsqrt_for_fma_fusion | 1.040× | 1.022× | ✓ | **✓ best** |
| D | fix_static_grid_to_use_dynamic_lambda_matching_autotuned_xblock | 0.979× | 0.959× | ✓ |  |
| E | add_triton_autotune | 1.014× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_broken_aut... | Δ | replace_int_div_s... | Δ | replace_int_div_s... | Δ | fix_static_grid_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7260 ns | -160 ns (-2.2%) | 7230 ns | -190 ns (-2.6%) | 7260 ns | -160 ns (-2.2%) | 7740 ns | +320 ns (+4.3%) | 7260 ns | -160 ns (-2.2%) |
| SM Throughput (%) | 38.23 % | 28.31 % | -9.92 pp | 27.58 % | -10.65 pp | 28.42 % | -9.81 pp | 38.32 % | +0.09 pp | 28.41 % | -9.82 pp |
| Memory Throughput (%) | 16.65 % | 17.64 % | +0.99 pp | 17.15 % | +0.50 pp | 17.70 % | +1.05 pp | 24.28 % | +7.63 pp | 17.70 % | +1.05 pp |
| DRAM Throughput (%) | 11.06 % | 11.34 % | +0.28 pp | 11.38 % | +0.32 pp | 11.33 % | +0.27 pp | 10.61 % | -0.45 pp | 11.35 % | +0.29 pp |
| Achieved Occupancy (%) | 57.16 % | 46.34 % | -10.82 pp | 46.07 % | -11.09 pp | 46.94 % | -10.22 pp | 43.10 % | -14.06 pp | 46.83 % | -10.33 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.71 % | 58.10 % | +15.39 pp | 58.01 % | +15.30 pp | 58.58 % | +15.87 pp | 39.65 % | -3.06 pp | 56.72 % | +14.01 pp |
| IPC Active | 2.25 inst/cycle | 1.69 inst/cycle | -0.56 | 1.68 inst/cycle | -0.57 | 1.72 inst/cycle | -0.53 | 2.39 inst/cycle | +0.14 | 1.71 inst/cycle | -0.54 |

---

## Iteration 4

**Best so far**: 1.022× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_at_fixed_xblock | 0.942× | 0.856× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency_with_correct_grid | 0.962× | 0.966× | ✓ |  |
| C | increase_num_warps_to_16_with_correct_lambda_grid_for_latency_hiding | 0.941× | 0.875× | ✓ |  |
| D | increase_num_warps_via_xblock256_fixed_config_to_hide_l1tex_latency | 1.016× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 8670 ns | +1250 ns (+16.8%) | 7680 ns | +260 ns (+3.5%) | 8480 ns | +1060 ns (+14.3%) | 7260 ns | -160 ns (-2.2%) |
| SM Throughput (%) | 38.23 % | 47.83 % | +9.60 pp | 52.59 % | +14.36 pp | 48.26 % | +10.03 pp | 28.39 % | -9.84 pp |
| Memory Throughput (%) | 16.65 % | 27.33 % | +10.68 pp | 29.81 % | +13.16 pp | 27.59 % | +10.94 pp | 17.74 % | +1.09 pp |
| DRAM Throughput (%) | 11.06 % | 9.48 % | -1.58 pp | 10.69 % | -0.37 pp | 9.67 % | -1.39 pp | 11.35 % | +0.29 pp |
| Achieved Occupancy (%) | 57.16 % | 74.10 % | +16.94 pp | 66.77 % | +9.61 pp | 73.90 % | +16.74 pp | 46.89 % | -10.27 pp |
| Registers/Thread | 17.00 register/thread | 19.00 register/thread | +2 | 18.00 register/thread | +1 | 19.00 register/thread | +2 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 7.42 | +3.71 | 7.42 | +3.71 | 7.42 | +3.71 | 3.71 | +0.00 |
| No Eligible (%) | 42.71 % | 34.43 % | -8.28 pp | 28.01 % | -14.70 pp | 34.87 % | -7.84 pp | 56.68 % | +13.97 pp |
| IPC Active | 2.25 inst/cycle | 2.56 inst/cycle | +0.31 | 2.87 inst/cycle | +0.62 | 2.60 inst/cycle | +0.35 | 1.71 inst/cycle | -0.54 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_to_hide_l1tex_latency_at_fixed_xblock` (0.856×)

Increasing block size from 128 to 256 threads (num_warps 4→8) on a pointwise kernel with a fixed grid size doubled the total executed instructions (1.63M → 3.26M) and doubled waves_per_sm (3.71 → 7.42), because the grid was not reduced to compensate for the larger block. Although occupancy improved (46% → 74%) and No-Eligible% dropped (58% → 34%), the Warp Cycles Per Issued Instruction remained essentially unchanged at ~17.9 cycles, indicating the L1TEX stall latency was not actually hidden — the bottleneck persisted and the extra instruction overhead increased duration from 7.23 µs to 8.67 µs. When the grid is fixed and block size doubles, each element is processed twice (once per original block assignment and once per new block), causing redundant work rather than latency hiding.

> **Applicable when**: Kernel shows high No-Eligible% (>50%), low eligible warps per scheduler (<1.5), and significant L1TEX stall fraction (>50% of CPI stall), with low achieved occupancy (<50%) and register headroom available (Block_Limit_Registers >> Block_Limit_Warps).

> **Avoid when**: The grid is fixed (not a dynamic lambda using meta['XBLOCK']). Increasing num_warps by doubling block size while keeping the grid constant doubles the total work dispatched, doubles instruction count, and doubles waves_per_sm — the latency-hiding benefit is negated by the extra work overhead. Also avoid when Warp Cycles Per Issued Instruction does not decrease after the change, as this signals the stall latency is not being hidden regardless of occupancy gains.

---

## Iteration 5

**Best so far**: 1.022× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_fma_fusion_and_evict_first_on_streaming_load | 1.014× | 1.022× | ✓ |  |
| B | fix_grid_lambda_to_eliminate_redundant_work_and_enable_fma_fusion | 0.958× | 0.934× | ✓ |  |
| C | rewrite_bn_arithmetic_as_fma_chain_with_correct_lambda_grid | 0.962× | 0.872× | ✓ |  |
| D | fix_static_grid_lambda_and_expand_autotune_configs_to_enable_correct_tile_selection | 1.039× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_fma_fusion... | Δ | fix_grid_lambda_t... | Δ | rewrite_bn_arithm... | Δ | fix_static_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 7260 ns | -160 ns (-2.2%) | 7940 ns | +520 ns (+7.0%) | 8510 ns | +1090 ns (+14.7%) | 7330 ns | -90 ns (-1.2%) |
| SM Throughput (%) | 38.23 % | 30.61 % | -7.62 pp | 51.49 % | +13.26 pp | 49.89 % | +11.66 pp | 13.98 % | -24.25 pp |
| Memory Throughput (%) | 16.65 % | 17.73 % | +1.08 pp | 29.54 % | +12.89 pp | 28.46 % | +11.81 pp | 8.41 % | -8.24 pp |
| DRAM Throughput (%) | 11.06 % | 11.33 % | +0.27 pp | 10.32 % | -0.74 pp | 9.68 % | -1.38 pp | 5.60 % | -5.46 pp |
| Achieved Occupancy (%) | 57.16 % | 47.22 % | -9.94 pp | 66.08 % | +8.92 pp | 73.34 % | +16.18 pp | 33.59 % | -23.57 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 19.00 register/thread | +2 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 7.42 | +3.71 | 7.42 | +3.71 | 1.86 | -1.85 |
| No Eligible (%) | 42.71 % | 54.47 % | +11.76 pp | 31.99 % | -10.72 pp | 34.18 % | -8.53 pp | 71.85 % | +29.14 pp |
| IPC Active | 2.25 inst/cycle | 1.82 inst/cycle | -0.43 | 2.82 inst/cycle | +0.57 | 2.60 inst/cycle | +0.35 | 1.09 inst/cycle | -1.16 |

---

## Cumulative Progress

```
Baseline:  7420 ns  (1.000×)
Iter 3     7260 ns  (1.022×)  — replace_int_div_sqrt_chain_with_rsqrt_for_fma_fusion

Final best: 7260 ns  (1.0220×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1003520
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
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
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp5 = tmp3 + tl.full([1], 1e-05, tl.float32)
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
+    tmp16 = tl.zeros([1], tl.float32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
