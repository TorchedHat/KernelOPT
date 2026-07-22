# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 408060 ns — **Best latency**: 405600 ns (**1.0061×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_enable_correct_xblock256_and_add_num_warps_2`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_3_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (658560,1,1), Waves/SM: 311.82
- **Registers/thread**: 17
- **Duration**: 408060 ns
- **SM Throughput**: 59.86 % | **Memory**: 32.31 % | **DRAM**: 32.31 %
- **Achieved Occupancy**: 60.94 %
- **IPC Active**: 2.41 inst/cycle | **No Eligible**: 40.16 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_hide_memory_latency | 0.678× | 0.672× | ✓ |  |
| B | autotune_xblock_increase_warp_latency_hiding | 0.999× | 0.999× | ✓ |  |
| C | vectorized_loads_stores_via_larger_xblock_autotune | 0.675× | 0.668× | ✓ |  |
| D | increase_xblock_to_hide_l1tex_latency | 1.000× | 0.509× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_i... | Δ | vectorized_loads_... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 408060 ns | 607620 ns | +199560 ns (+48.9%) | 408380 ns | +320 ns (+0.1%) | 611260 ns | +203200 ns (+49.8%) | 801920 ns | +393860 ns (+96.5%) |
| SM Throughput (%) | 59.86 % | 87.29 % | +27.43 pp | 70.05 % | +10.19 pp | 87.16 % | +27.30 pp | 85.85 % | +25.99 pp |
| Memory Throughput (%) | 32.31 % | 25.74 % | -6.57 pp | 32.29 % | -0.02 pp | 25.69 % | -6.62 pp | 27.40 % | -4.91 pp |
| DRAM Throughput (%) | 32.31 % | 21.70 % | -10.61 pp | 32.29 % | -0.02 pp | 21.57 % | -10.74 pp | 16.44 % | -15.87 pp |
| Achieved Occupancy (%) | 60.94 % | 84.19 % | +23.25 pp | 64.98 % | +4.04 pp | 84.54 % | +23.60 pp | 78.71 % | +17.77 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 24.00 register/thread | +7 |
| Waves/SM | 311.82 | 311.82 | +0.00 | 311.82 | +0.00 | 311.82 | +0.00 | 623.64 | +311.82 |
| No Eligible (%) | 40.16 % | 12.53 % | -27.63 pp | 29.53 % | -10.63 pp | 12.50 % | -27.66 pp | 13.84 % | -26.32 pp |
| IPC Active | 2.41 inst/cycle | 3.50 inst/cycle | +1.09 | 2.83 inst/cycle | +0.42 | 3.50 inst/cycle | +1.09 | 3.45 inst/cycle | +1.04 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_hide_memory_latency` (0.672×)

Adding triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate significantly more registers per thread (17→32), which reduced Block Limit Registers from 21 to 16 and more than doubled the total executed instruction count (192M→414M). Although eligible warps per scheduler improved (1.13→6.79) and achieved occupancy rose (61%→84%), the 2.15x instruction count increase dominated, extending kernel duration from 408µs to 608µs. The autotune overhead also introduced a hardcoded grid that no longer matched the autotuned XBLOCK, meaning the grid was not updated to use a lambda, so the kernel may have launched with a mismatched grid. The structural anti-pattern is: adding autotune configs with larger XBLOCK when the kernel body is already register-efficient at the baseline tile size can trigger the compiler to emit more instructions (e.g., unrolled loops, additional predicate logic from switching xmask from tl.full to xindex < xnumel), negating the latency-hiding benefit.

> **Applicable when**: Never: this specific combination of autotune + xmask change + static grid caused regression. Avoid this pattern.

> **Avoid when**: When the baseline kernel uses tl.full([XBLOCK], True, tl.int1) for xmask (indicating no boundary masking needed), do NOT switch to xindex < xnumel when adding autotune — this adds predicate overhead. Also do NOT use a static grid when adding autotune; the grid must be a lambda using META['XBLOCK']. When registers per thread are already low (≤20) but the autotune configs cause the compiler to double register usage, the occupancy gain is outweighed by instruction count explosion.

#### ❌ Plan D: `increase_xblock_to_hide_l1tex_latency` (0.509×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel with a fixed grid (not updated to use META['XBLOCK']) caused the grid to remain at cdiv(N, 128) while the kernel processed XBLOCK=256 elements per thread block, resulting in ~2.8x more total executed instructions (192M → 540M) and nearly doubling kernel duration (408µs → 802µs). The autotune selected a larger block size but the launch grid was not updated to use META['XBLOCK'], so each element was processed multiple times. Even when the grid is correctly parameterized, autotune overhead on a kernel with 311+ waves/SM (already massively over-subscribed) adds tuning cost without proportional benefit, and larger blocks increased register pressure (17→24 regs/thread), reducing the block limit from 21 to 10 blocks/SM.

> **Applicable when**: Never apply autotune with larger XBLOCK without also updating the grid lambda to use META['XBLOCK']. The implementation flaw (fixed grid) is the primary cause of regression here.

> **Avoid when**: Do not apply autotune with larger XBLOCK configs when: (1) the grid is not updated to use META['XBLOCK'] — this causes redundant work and instruction explosion; (2) the kernel already has 300+ waves/SM (massively over-subscribed), where occupancy improvements from larger blocks yield diminishing returns; (3) register pressure increase from larger blocks reduces block limit significantly (e.g., 21→10 blocks/SM).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_enable_fma_via_reciprocal_multiply | 1.002× | 1.002× | ✓ |  |
| B | fuse_fp32_instructions_via_rsqrt_and_fma_rewrite | 0.886× | 0.888× | ✓ |  |
| C | replace_nonfused_fp32_div_sequence_with_rsqrt_fma | 0.886× | 0.892× | ✓ |  |
| D | replace_integer_reciprocal_sqrt_with_rsqrt_to_enable_fma_fusion | 1.002× | 0.763× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fuse_fp32_instruc... | Δ | replace_nonfused_... | Δ | replace_integer_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 408060 ns | 407070 ns | -990 ns (-0.2%) | 459330 ns | +51270 ns (+12.6%) | 457310 ns | +49250 ns (+12.1%) | 534850 ns | +126790 ns (+31.1%) | 408480 ns | +420 ns (+0.1%) |
| SM Throughput (%) | 59.86 % | 42.71 % | -17.15 pp | 53.54 % | -6.32 pp | 53.33 % | -6.53 pp | 74.48 % | +14.62 pp | 70.07 % | +10.21 pp |
| Memory Throughput (%) | 32.31 % | 32.39 % | +0.08 pp | 34.62 % | +2.31 pp | 34.50 % | +2.19 pp | 42.48 % | +10.17 pp | 32.28 % | -0.03 pp |
| DRAM Throughput (%) | 32.31 % | 32.39 % | +0.08 pp | 28.73 % | -3.58 pp | 28.86 % | -3.45 pp | 24.66 % | -7.65 pp | 32.28 % | -0.03 pp |
| Achieved Occupancy (%) | 60.94 % | 45.99 % | -14.95 pp | 36.44 % | -24.50 pp | 36.80 % | -24.14 pp | 72.57 % | +11.63 pp | 64.66 % | +3.72 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 311.82 | 311.82 | +0.00 | 311.82 | +0.00 | 311.82 | +0.00 | 623.64 | +311.82 | 311.82 | +0.00 |
| No Eligible (%) | 40.16 % | 57.04 % | +16.88 pp | 46.22 % | +6.06 pp | 45.96 % | +5.80 pp | 25.35 % | -14.81 pp | 29.45 % | -10.71 pp |
| IPC Active | 2.41 inst/cycle | 1.71 inst/cycle | -0.70 | 2.18 inst/cycle | -0.23 | 2.15 inst/cycle | -0.26 | 2.99 inst/cycle | +0.58 | 2.82 inst/cycle | +0.41 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_to_enable_correct_tile_exploration | 1.002× | 1.004× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_correct_xblock_selection | 1.003× | 1.003× | ✓ | **✓ best** |
| C | fix_broken_autotune_grid_lambda_and_tune_num_warps_for_l1tex_latency_hiding | 1.002× | 1.002× | ✓ |  |
| D | revert_autotune_configs_to_small_xblock_restore_baseline_efficiency | 1.003× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | fix_autotune_grid... | Δ | fix_broken_autotu... | Δ | revert_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 408060 ns | 406620 ns | -1440 ns (-0.4%) | 406940 ns | -1120 ns (-0.3%) | 407140 ns | -920 ns (-0.2%) | 407040 ns | -1020 ns (-0.2%) |
| SM Throughput (%) | 59.86 % | 42.48 % | -17.38 pp | 42.52 % | -17.34 pp | 42.54 % | -17.32 pp | 42.79 % | -17.07 pp |
| Memory Throughput (%) | 32.31 % | 32.42 % | +0.11 pp | 32.41 % | +0.10 pp | 32.39 % | +0.08 pp | 32.38 % | +0.07 pp |
| DRAM Throughput (%) | 32.31 % | 32.42 % | +0.11 pp | 32.41 % | +0.10 pp | 32.39 % | +0.08 pp | 32.38 % | +0.07 pp |
| Achieved Occupancy (%) | 60.94 % | 46.02 % | -14.92 pp | 45.90 % | -15.04 pp | 46.00 % | -14.94 pp | 46.14 % | -14.80 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 311.82 | 311.82 | +0.00 | 311.82 | +0.00 | 311.82 | +0.00 | 311.82 | +0.00 |
| No Eligible (%) | 40.16 % | 57.29 % | +17.13 pp | 57.21 % | +17.05 pp | 57.23 % | +17.07 pp | 57.05 % | +16.89 pp |
| IPC Active | 2.41 inst/cycle | 1.71 inst/cycle | -0.70 | 1.71 inst/cycle | -0.70 | 1.71 inst/cycle | -0.70 | 1.72 inst/cycle | -0.69 |

---

## Iteration 4

**Best so far**: 1.0028× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_streaming_load_to_protect_channel_vector_cache_residency | 1.002× | 1.003× | ✓ |  |
| B | streaming_eviction_policy_for_large_tensor_plus_fix_broken_autotune_grid | 1.002× | 1.003× | ✓ |  |
| C | fix_autotune_grid_lambda_and_tune_num_warps_for_warp_latency_hiding | 1.002× | 1.003× | ✓ |  |
| D | increase_num_stages_to_pipeline_l1tex_loads | 1.000× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | streaming_evictio... | Δ | fix_autotune_grid... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 408060 ns | 406720 ns | -1340 ns (-0.3%) | 406980 ns | -1080 ns (-0.3%) | 406940 ns | -1120 ns (-0.3%) | 407100 ns | -960 ns (-0.2%) |
| SM Throughput (%) | 59.86 % | 46.05 % | -13.81 pp | 46.56 % | -13.30 pp | 42.47 % | -17.39 pp | 42.49 % | -17.37 pp |
| Memory Throughput (%) | 32.31 % | 32.43 % | +0.12 pp | 32.39 % | +0.08 pp | 32.41 % | +0.10 pp | 32.38 % | +0.07 pp |
| DRAM Throughput (%) | 32.31 % | 32.43 % | +0.12 pp | 32.39 % | +0.08 pp | 32.41 % | +0.10 pp | 32.38 % | +0.07 pp |
| Achieved Occupancy (%) | 60.94 % | 46.49 % | -14.45 pp | 46.57 % | -14.37 pp | 45.99 % | -14.95 pp | 46.07 % | -14.87 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 311.82 | 311.82 | +0.00 | 311.82 | +0.00 | 311.82 | +0.00 | 311.82 | +0.00 |
| No Eligible (%) | 40.16 % | 53.95 % | +13.79 pp | 53.11 % | +12.95 pp | 57.21 % | +17.05 pp | 57.20 % | +17.04 pp |
| IPC Active | 2.41 inst/cycle | 1.84 inst/cycle | -0.57 | 1.87 inst/cycle | -0.54 | 1.71 inst/cycle | -0.70 | 1.72 inst/cycle | -0.69 |

---

## Iteration 5

**Best so far**: 1.0028× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_fixed_xblock_for_l1tex_latency_hiding | 1.002× | 1.002× | ✓ |  |
| B | fix_autotune_grid_lambda_enable_correct_xblock256_and_add_num_warps_2 | 1.007× | 1.006× | ✓ | **✓ best** |
| C | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.007× | 1.006× | ✓ |  |
| D | fix_broken_autotune_grid_lambda_and_add_num_warps_16_for_latency_hiding | 1.002× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_autotune_grid... | Δ | increase_num_warp... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 408060 ns | 407100 ns | -960 ns (-0.2%) | 405600 ns | -2460 ns (-0.6%) | 405570 ns | -2490 ns (-0.6%) | 406620 ns | -1440 ns (-0.4%) |
| SM Throughput (%) | 59.86 % | 46.00 % | -13.86 pp | 25.01 % | -34.85 pp | 27.06 % | -32.80 pp | 42.48 % | -17.38 pp |
| Memory Throughput (%) | 32.31 % | 32.39 % | +0.08 pp | 32.52 % | +0.21 pp | 32.52 % | +0.21 pp | 32.43 % | +0.12 pp |
| DRAM Throughput (%) | 32.31 % | 32.39 % | +0.08 pp | 32.52 % | +0.21 pp | 32.52 % | +0.21 pp | 32.43 % | +0.12 pp |
| Achieved Occupancy (%) | 60.94 % | 46.45 % | -14.49 pp | 26.91 % | -34.03 pp | 27.21 % | -33.73 pp | 45.90 % | -15.04 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 311.82 | 311.82 | +0.00 | 155.91 | -155.91 | 155.91 | -155.91 | 311.82 | +0.00 |
| No Eligible (%) | 40.16 % | 53.69 % | +13.53 pp | 74.76 % | +34.60 pp | 72.67 % | +32.51 pp | 57.21 % | +17.05 pp |
| IPC Active | 2.41 inst/cycle | 1.84 inst/cycle | -0.57 | 1.01 inst/cycle | -1.40 | 1.09 inst/cycle | -1.32 | 1.71 inst/cycle | -0.70 |

---

## Cumulative Progress

```
Baseline:  408060 ns  (1.000×)
Iter 3     406940 ns  (1.003×)  — fix_autotune_grid_lambda_to_enable_correct_xblock_selection
Iter 5     405600 ns  (1.006×)  — fix_autotune_grid_lambda_enable_correct_xblock256_and_add_num_warps_2

Final best: 405600 ns  (1.0061×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 84295680
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 672)
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
+    inv_std = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * inv_std
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
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
@@ -45,7 +51,7 @@

     arg_4 = inputs[4]
     xnumel = 84295680
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 84295680, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 84295680)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
