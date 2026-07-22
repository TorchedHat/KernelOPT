# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42980 ns — **Best latency**: 42940 ns (**1.0009×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_broken_persistent_loop_to_simple_pointwise`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_039/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 42980 ns
- **SM Throughput**: 61.91 % | **Memory**: 31.30 % | **DRAM**: 19.41 %
- **Achieved Occupancy**: 57.63 %
- **IPC Active**: 2.59 inst/cycle | **No Eligible**: 34.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_occupancy_and_latency_hiding | 0.614× | 0.599× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.992× | 0.996× | ✓ |  |
| C | autotune_xblock_num_warps_for_occupancy_and_l1_reuse | 0.991× | 0.175× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_better_occupancy | 0.612× | 0.599× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_t... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42980 ns | 71740 ns | +28760 ns (+66.9%) | 43170 ns | +190 ns (+0.4%) | 245500 ns | +202520 ns (+471.2%) | 71740 ns | +28760 ns (+66.9%) |
| SM Throughput (%) | 61.91 % | 83.31 % | +21.40 pp | 76.51 % | +14.60 pp | 85.74 % | +23.83 pp | 83.14 % | +21.23 pp |
| Memory Throughput (%) | 31.30 % | 28.53 % | -2.77 pp | 31.21 % | -0.09 pp | 20.90 % | -10.40 pp | 28.45 % | -2.85 pp |
| DRAM Throughput (%) | 19.41 % | 11.57 % | -7.84 pp | 19.39 % | -0.02 pp | 3.41 % | -16.00 pp | 11.59 % | -7.82 pp |
| Achieved Occupancy (%) | 57.63 % | 89.14 % | +31.51 pp | 73.21 % | +15.58 pp | 44.56 % | -13.07 pp | 89.09 % | +31.46 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 52.00 register/thread | +30 | 32.00 register/thread | +10 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 124.12 | +93.09 | 31.03 | +0.00 |
| No Eligible (%) | 34.50 % | 14.49 % | -20.01 pp | 19.25 % | -15.25 pp | 13.72 % | -20.78 pp | 14.25 % | -20.25 pp |
| IPC Active | 2.59 inst/cycle | 3.42 inst/cycle | +0.83 | 3.23 inst/cycle | +0.64 | 3.46 inst/cycle | +0.87 | 3.42 inst/cycle | +0.83 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_tile_size_for_occupancy_and_latency_hiding` (0.599×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) successfully raised achieved occupancy from 57.6% to 89.1% and eligible warps/scheduler from 1.37 to 7.45, but caused a 2.24x increase in executed instructions (20.97M → 46.92M) and a 67% increase in kernel duration (42.98 µs → 71.74 µs). The instruction explosion likely stems from the xmask guard (xindex < xnumel) being evaluated per-element at larger tile sizes, combined with the autotune overhead selecting a configuration that generates significantly more predicated/masked instructions. When the original kernel used tl.full([XBLOCK], True, ...) with a fixed XBLOCK that evenly divides xnumel, no masking overhead existed; replacing this with a dynamic xmask at larger tile sizes multiplies instruction count even when xnumel is divisible by all candidate XBLOCK values.

> **Applicable when**: Never apply this specific pattern (autotune larger XBLOCK + dynamic xmask replacement) when the original kernel uses tl.full True mask (no masking) and xnumel is already evenly divisible by the fixed XBLOCK, as the masking overhead dominates any occupancy gains.

> **Avoid when**: Do not replace tl.full([XBLOCK], True, tl.int1) with dynamic xmask = xindex < xnumel when adding autotune with larger XBLOCK configs, if xnumel is divisible by all candidate XBLOCK values — the mask is unnecessary and adds significant instruction overhead. Also avoid when the grid lambda is not updated to use meta['XBLOCK'], causing the grid to remain fixed at the original small tile size while the kernel uses larger tiles.

#### ❌ Plan D: `increase_xblock_for_vectorized_loads_and_better_occupancy` (0.599×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) on a pointwise kernel caused a 1.67x duration increase despite improving achieved occupancy from 57.6% to 89.1%. The critical failure was that executed instructions ballooned from ~21M to ~47M (2.24x), driven by the introduction of a non-trivial xmask (xindex < xnumel) that replaced tl.full([XBLOCK], True) — this added per-element predicate computation overhead across all loads and stores. Additionally, registers per thread increased from 22 to 32, reducing the block limit from 21 to 16 blocks/SM, which partially offset the occupancy gains. When the original kernel already uses tl.full([XBLOCK], True) as its mask (meaning xnumel is exactly divisible by XBLOCK), switching to a dynamic mask introduces unnecessary predicate instructions that dominate the compute cost.

> **Applicable when**: Never — this specific combination of adding dynamic masking while autotuning XBLOCK is an anti-pattern. Autotuning XBLOCK alone (without changing the mask) may be valid when the original kernel already uses a dynamic mask.

> **Avoid when**: When the original kernel uses tl.full([XBLOCK], True) as its mask (indicating xnumel is exactly divisible by XBLOCK for all candidate configs), do NOT replace it with a dynamic xindex < xnumel mask — this adds unnecessary predicate computation that inflates instruction count by 2x+. Also, the grid must be updated to a lambda using meta['XBLOCK'] when autotuning; using a static grid with a fixed XBLOCK defeats the purpose of autotuning.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_integer_div_rsqrt_with_fused_rsqrt_to_eliminate_nonfused_fp_instructions | 1.002× | 1.001× | ✓ | **✓ best** |
| B | persistent_tile_scheduling_with_mask_regression_fix | 0.013× | 0.012× | ✓ |  |
| C | replace_integer_reciprocal_sqrt_with_rsqrt_to_enable_fma_fusion | 0.894× | 0.904× | ✓ |  |
| D | persistent_tile_scheduling_with_register_pressure_reduction | 1.001× | 0.999× | ✓ |  |
| E | add_triton_autotune | 0.993× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_integer_d... | Δ | persistent_tile_s... | Δ | replace_integer_r... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42980 ns | 42940 ns | -40 ns (-0.1%) | 3480000 ns | +3437020 ns (+7996.8%) | 47520 ns | +4540 ns (+10.6%) | 43010 ns | +30 ns (+0.1%) | 43490 ns | +510 ns (+1.2%) |
| SM Throughput (%) | 61.91 % | 51.04 % | -10.87 pp | 81.16 % | +19.25 pp | 66.26 % | +4.35 pp | 55.76 % | -6.15 pp | 76.94 % | +15.03 pp |
| Memory Throughput (%) | 31.30 % | 31.22 % | -0.08 pp | 90.33 % | +59.03 pp | 44.24 % | +12.94 pp | 30.83 % | -0.47 pp | 31.39 % | +0.09 pp |
| DRAM Throughput (%) | 19.41 % | 19.38 % | -0.03 pp | 1.90 % | -17.51 pp | 17.50 % | -1.91 pp | 19.33 % | -0.08 pp | 19.11 % | -0.30 pp |
| Achieved Occupancy (%) | 57.63 % | 43.83 % | -13.80 pp | 98.64 % | +41.01 pp | 46.47 % | -11.16 pp | 44.77 % | -12.86 pp | 72.48 % | +14.85 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 24.00 register/thread | +2 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 34.50 % | 46.30 % | +11.80 pp | 18.79 % | -15.71 pp | 29.45 % | -5.05 pp | 52.62 % | +18.12 pp | 19.86 % | -14.64 pp |
| IPC Active | 2.59 inst/cycle | 2.16 inst/cycle | -0.43 | 3.25 inst/cycle | +0.66 | 2.82 inst/cycle | +0.23 | 1.92 inst/cycle | -0.67 | 3.21 inst/cycle | +0.62 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling_with_mask_regression_fix` (0.716×)

Adding a persistent while-loop (tile_id += num_sms) while keeping the original large grid (65,536 CTAs) instead of reducing it to (NUM_SMS,) caused a 1.82x instruction explosion (17.3M → 31.5M) and a 40% duration increase (43 µs → 60 µs). Although occupancy improved (43.8% → 78.1%) and eligible warps/scheduler rose (0.92 → 2.74), the warp cycles per issued instruction worsened (12.59 → 17.53), indicating the persistent loop added overhead without reducing stalls. The grid must be set to exactly (NUM_SMS,) when using a persistent tile loop; leaving it at (triton.cdiv(xnumel, XBLOCK),) causes each CTA to process multiple tiles sequentially, multiplying total instruction count by the loop iteration count.

> **Applicable when**: Never — this is a negative result. Do not apply persistent while-loop scheduling without simultaneously reducing the grid to (NUM_SMS,).

> **Avoid when**: Do not add a persistent tile while-loop when the grid is not simultaneously reduced to (NUM_SMS,). When tl.num_programs(0) returns the full grid size (e.g., 65536) instead of NUM_SMS (132), the stride in tile_id += num_sms is too large, causing each CTA to process only ~1 tile anyway but with the overhead of the loop construct and doubled register pressure (22 → 32 regs/thread). Also avoid persistent kernels when waves_per_sm > 10 and the workload already saturates the GPU.

---

## Iteration 3

**Best so far**: 1.0009× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_policy_for_large_non_reused_tensor | 1.000× | 1.000× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_stall_latency | 0.717× | 0.716× | ✓ |  |
| C | fma_fusion_via_constexpr_epsilon_and_arithmetic_reorder | 0.892× | 0.904× | ✓ |  |
| D | fix_persistent_loop_grid_mismatch_and_revert_to_standard_pointwise | 0.999× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | increase_num_warp... | Δ | fma_fusion_via_co... | Δ | fix_persistent_lo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42980 ns | 42980 ns | +0 ns (+0.0%) | 60000 ns | +17020 ns (+39.6%) | 47550 ns | +4570 ns (+10.6%) | 43100 ns | +120 ns (+0.3%) |
| SM Throughput (%) | 61.91 % | 56.03 % | -5.88 pp | 80.38 % | +18.47 pp | 66.24 % | +4.33 pp | 49.34 % | -12.57 pp |
| Memory Throughput (%) | 31.30 % | 31.34 % | +0.04 pp | 44.01 % | +12.71 pp | 44.27 % | +12.97 pp | 30.78 % | -0.52 pp |
| DRAM Throughput (%) | 19.41 % | 19.54 % | +0.13 pp | 13.93 % | -5.48 pp | 17.57 % | -1.84 pp | 19.32 % | -0.09 pp |
| Achieved Occupancy (%) | 57.63 % | 44.60 % | -13.03 pp | 78.08 % | +20.45 pp | 46.27 % | -11.36 pp | 43.86 % | -13.77 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 18.00 register/thread | -4 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 62.06 | +31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 34.50 % | 41.32 % | +6.82 pp | 30.22 % | -4.28 pp | 29.58 % | -4.92 pp | 60.44 % | +25.94 pp |
| IPC Active | 2.59 inst/cycle | 2.36 inst/cycle | -0.23 | 2.79 inst/cycle | +0.20 | 2.82 inst/cycle | +0.23 | 1.56 inst/cycle | -1.03 |

---

## Iteration 4

**Best so far**: 1.0009× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | evict_first_streaming_load_and_store_for_main_tensor_to_improve_parameter_cache_reuse | 1.002× | 1.000× | ✓ |  |
| C | replace_modulo_with_bitwise_and_to_eliminate_integer_div_and_enable_fma_fusion | 0.493× | 0.472× | ✓ |  |
| D | revert_broken_persistent_loop_to_simple_pointwise | 1.005× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | replace_modulo_wi... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42980 ns | 42980 ns | +0 ns (+0.0%) | 91010 ns | +48030 ns (+111.7%) | 42940 ns | -40 ns (-0.1%) |
| SM Throughput (%) | 61.91 % | 49.05 % | -12.86 pp | 60.99 % | -0.92 pp | 49.65 % | -12.26 pp |
| Memory Throughput (%) | 31.30 % | 30.55 % | -0.75 pp | 69.00 % | +37.70 pp | 31.09 % | -0.21 pp |
| DRAM Throughput (%) | 19.41 % | 19.38 % | -0.03 pp | 9.25 % | -10.16 pp | 19.38 % | -0.03 pp |
| Achieved Occupancy (%) | 57.63 % | 43.11 % | -14.52 pp | 90.57 % | +32.94 pp | 41.72 % | -15.91 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 34.50 % | 57.59 % | +23.09 pp | 37.73 % | +3.23 pp | 60.85 % | +26.35 pp |
| IPC Active | 2.59 inst/cycle | 1.69 inst/cycle | -0.90 | 2.49 inst/cycle | -0.10 | 1.57 inst/cycle | -1.02 |

---

## Iteration 5

**Best so far**: 1.0009× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_num_stages_to_hide_l1tex_latency | 0.998× | 1.002× | ✓ |  |
| C | revert_to_coalesced_xblock128_with_static_mask_and_evict_first | 1.004× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | revert_to_coalesc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42980 ns | 42910 ns | -70 ns (-0.2%) | 42910 ns | -70 ns (-0.2%) |
| SM Throughput (%) | 61.91 % | 49.42 % | -12.49 pp | 49.67 % | -12.24 pp |
| Memory Throughput (%) | 31.30 % | 30.95 % | -0.35 pp | 30.99 % | -0.31 pp |
| DRAM Throughput (%) | 19.41 % | 19.36 % | -0.05 pp | 19.39 % | -0.02 pp |
| Achieved Occupancy (%) | 57.63 % | 41.54 % | -16.09 pp | 41.21 % | -16.42 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 34.50 % | 60.55 % | +26.05 pp | 59.47 % | +24.97 pp |
| IPC Active | 2.59 inst/cycle | 1.58 inst/cycle | -1.01 | 1.62 inst/cycle | -0.97 |

---

## Cumulative Progress

```
Baseline:  42980 ns  (1.000×)
Iter 2     42940 ns  (1.001×)  — replace_integer_div_rsqrt_with_fused_rsqrt_to_eliminate_nonfused_fp_instructions

Final best: 42940 ns  (1.0009×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1)],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 8192)
+    x0 = xindex % 8192
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tmp4 = tmp2 * tmp3
-    tmp6 = tmp4 - tmp5
-    tmp8 = tl.full([1], 1e-05, tl.float32)
-    tmp9 = tmp7 + tmp8
-    tmp10 = tl.sqrt_rn(tmp9)
-    tmp11 = tl.full([1], 1, tl.int32)
-    tmp12 = (tmp11 / tmp10)
-    tmp13 = tl.full([1], 1.0, tl.float32)
-    tmp14 = tmp12 * tmp13
+    tmp0  = tl.load(in_ptr0     + x0, eviction_policy='evict_last')
+    tmp1  = tl.load(in_out_ptr0 + x2)
+    tmp3  = tl.load(in_ptr1     + x0, eviction_policy='evict_last')
+    tmp5  = tl.load(in_ptr2     + x0, eviction_policy='evict_last')
+    tmp7  = tl.load(in_ptr3     + x0, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4     + x0, eviction_policy='evict_last')
+    tmp18 = tl.load(in_ptr5     + x0, eviction_policy='evict_last')
+    tmp2  = tmp0 + tmp1
+    tmp4  = tmp2 * tmp3
+    tmp6  = tmp4 - tmp5
+    tmp8  = tl.full([1], 1e-05, tl.float32)
+    tmp9  = tmp7 + tmp8
+    tmp14 = tl.math.rsqrt(tmp9)
     tmp15 = tmp6 * tmp14
     tmp17 = tmp15 * tmp16
     tmp19 = tmp17 + tmp18
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + x2, tmp19)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -47,7 +44,7 @@

     arg_6 = inputs[6]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
