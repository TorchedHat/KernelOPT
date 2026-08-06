# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_64` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8640 ns — **Best latency**: 7940 ns (**1.0882×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_num_warps_to_1_to_minimize_instruction_count`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_64`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_64_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8085,1,1), Waves/SM: 3.83
- **Registers/thread**: 22
- **Duration**: 8640 ns
- **SM Throughput**: 55.71 % | **Memory**: 39.22 % | **DRAM**: 9.79 %
- **Achieved Occupancy**: 77.51 %
- **IPC Active**: 2.73 inst/cycle | **No Eligible**: 29.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 1.000× | 0.252× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 1.003× | 0.954× | ✓ | **✓ best** |
| C | increase_xblock_vectorized_loads_autotune | 0.811× | 0.798× | ✓ |  |
| D | eliminate_partial_wave_via_autotune_xblock | 0.958× | 0.954× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_v... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8640 ns | 34300 ns | +25660 ns (+297.0%) | 9060 ns | +420 ns (+4.9%) | 10820 ns | +2180 ns (+25.2%) | 9060 ns | +420 ns (+4.9%) |
| SM Throughput (%) | 55.71 % | 77.10 % | +21.39 pp | 54.66 % | -1.05 pp | 62.25 % | +6.54 pp | 60.29 % | +4.58 pp |
| Memory Throughput (%) | 39.22 % | 24.18 % | -15.04 pp | 36.10 % | -3.12 pp | 31.68 % | -7.54 pp | 33.15 % | -6.07 pp |
| DRAM Throughput (%) | 9.79 % | 2.46 % | -7.33 pp | 9.35 % | -0.44 pp | 7.83 % | -1.96 pp | 9.35 % | -0.44 pp |
| Achieved Occupancy (%) | 77.51 % | 87.79 % | +10.28 pp | 79.39 % | +1.88 pp | 81.34 % | +3.83 pp | 77.30 % | -0.21 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.83 | 15.31 | +11.48 | 3.83 | +0.00 | 3.83 | +0.00 | 3.83 | +0.00 |
| No Eligible (%) | 29.82 % | 19.73 % | -10.09 pp | 31.32 % | +1.50 pp | 24.34 % | -5.48 pp | 24.69 % | -5.13 pp |
| IPC Active | 2.73 inst/cycle | 3.21 inst/cycle | +0.48 | 2.71 inst/cycle | -0.02 | 2.97 inst/cycle | +0.24 | 2.95 inst/cycle | +0.22 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_alu_latency | 0.975× | 0.575× | ✓ |  |
| B | fix_autotune_grid_lambda_and_add_low_warp_configs | 1.077× | 1.121× | ✓ | **✓ best** |
| C | revert_harmful_autotune_restore_baseline_xblock128 | 0.999× | 1.004× | ✓ |  |
| D | revert_autotune_and_cap_registers_to_restore_occupancy | 0.979× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_autotune_grid... | Δ | revert_harmful_au... | Δ | revert_autotune_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8640 ns | 15040 ns | +6400 ns (+74.1%) | 7710 ns | -930 ns (-10.8%) | 8610 ns | -30 ns (-0.3%) | 8700 ns | +60 ns (+0.7%) |
| SM Throughput (%) | 55.71 % | 64.54 % | +8.83 pp | 34.53 % | -21.18 pp | 55.33 % | -0.38 pp | 54.02 % | -1.69 pp |
| Memory Throughput (%) | 39.22 % | 25.84 % | -13.38 pp | 38.93 % | -0.29 pp | 39.48 % | +0.26 pp | 39.15 % | -0.07 pp |
| DRAM Throughput (%) | 9.79 % | 5.62 % | -4.17 pp | 10.95 % | +1.16 pp | 9.83 % | +0.04 pp | 9.72 % | -0.07 pp |
| Achieved Occupancy (%) | 77.51 % | 84.49 % | +6.98 pp | 61.34 % | -16.17 pp | 79.07 % | +1.56 pp | 77.66 % | +0.15 pp |
| Registers/Thread | 22.00 register/thread | 23.00 register/thread | +1 | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 3.83 | 7.66 | +3.83 | 1.91 | -1.92 | 3.83 | +0.00 | 3.83 | +0.00 |
| No Eligible (%) | 29.82 % | 28.86 % | -0.96 pp | 38.89 % | +9.07 pp | 31.98 % | +2.16 pp | 30.89 % | +1.07 pp |
| IPC Active | 2.73 inst/cycle | 2.83 inst/cycle | +0.10 | 2.46 inst/cycle | -0.27 | 2.61 inst/cycle | -0.12 | 2.71 inst/cycle | -0.02 |

---

## Iteration 3

**Best so far**: 1.1206× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | force_num_warps4_at_xblock128_to_hide_l1tex_latency | 0.979× | 0.993× | ✓ |  |
| B | add_num_warps_2_autotune_config_to_reduce_instruction_count | 1.077× | 1.116× | ✓ |  |
| C | add_low_num_warps_autotune_config_to_reduce_instruction_count | 1.058× | 1.084× | ✓ |  |
| D | add_low_num_warps_autotune_configs_to_reduce_cpi_stall_and_instruction_count | 1.077× | 1.088× | ✓ |  |


### Metric Comparison

| Metric | Baseline | force_num_warps4_... | Δ | add_num_warps_2_a... | Δ | add_low_num_warps... | Δ | add_low_num_warps... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8640 ns | 8700 ns | +60 ns (+0.7%) | 7740 ns | -900 ns (-10.4%) | 7970 ns | -670 ns (-7.8%) | 7940 ns | -700 ns (-8.1%) |
| SM Throughput (%) | 55.71 % | 54.52 % | -1.19 pp | 34.59 % | -21.12 pp | 34.41 % | -21.30 pp | 21.00 % | -34.71 pp |
| Memory Throughput (%) | 39.22 % | 38.98 % | -0.24 pp | 38.94 % | -0.28 pp | 37.60 % | -1.62 pp | 33.59 % | -5.63 pp |
| DRAM Throughput (%) | 9.79 % | 9.70 % | -0.09 pp | 10.97 % | +1.18 pp | 10.63 % | +0.84 pp | 10.70 % | +0.91 pp |
| Achieved Occupancy (%) | 77.51 % | 79.30 % | +1.79 pp | 59.11 % | -18.40 pp | 64.14 % | -13.37 pp | 25.72 % | -51.79 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 50.00 register/thread | +28 |
| Waves/SM | 3.83 | 3.83 | +0.00 | 1.91 | -1.92 | 1.91 | -1.92 | 1.91 | -1.92 |
| No Eligible (%) | 29.82 % | 30.35 % | +0.53 pp | 35.62 % | +5.80 pp | 36.59 % | +6.77 pp | 61.57 % | +31.75 pp |
| IPC Active | 2.73 inst/cycle | 2.76 inst/cycle | +0.03 | 2.40 inst/cycle | -0.33 | 2.35 inst/cycle | -0.38 | 1.46 inst/cycle | -1.27 |

---

## Iteration 4

**Best so far**: 1.1206× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_last_cache_hint_to_reduce_l1tex_stall | 1.091× | 1.076× | ✓ | **✓ best** |
| B | add_evict_last_cache_hint_to_reduce_l1tex_stall | 1.090× | 1.088× | ✓ |  |
| C | add_eviction_policy_hints_to_reduce_l1tex_stalls | 1.072× | 1.088× | ✓ |  |
| D | add_num_warps_2_config_and_fix_grid_lambda_to_halve_instruction_count | 1.076× | 1.098× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_last_ca... | Δ | add_evict_last_ca... | Δ | add_eviction_poli... | Δ | add_num_warps_2_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8640 ns | 8030 ns | -610 ns (-7.1%) | 7940 ns | -700 ns (-8.1%) | 7940 ns | -700 ns (-8.1%) | 7870 ns | -770 ns (-8.9%) |
| SM Throughput (%) | 55.71 % | 40.63 % | -15.08 pp | 24.29 % | -31.42 pp | 42.03 % | -13.68 pp | 34.68 % | -21.03 pp |
| Memory Throughput (%) | 39.22 % | 37.72 % | -1.50 pp | 33.41 % | -5.81 pp | 38.27 % | -0.95 pp | 37.88 % | -1.34 pp |
| DRAM Throughput (%) | 9.79 % | 10.56 % | +0.77 pp | 10.69 % | +0.90 pp | 10.67 % | +0.88 pp | 10.79 % | +1.00 pp |
| Achieved Occupancy (%) | 77.51 % | 62.75 % | -14.76 pp | 25.97 % | -51.54 pp | 63.18 % | -14.33 pp | 61.87 % | -15.64 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 50.00 register/thread | +28 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.83 | 1.91 | -1.92 | 1.91 | -1.92 | 1.91 | -1.92 | 1.91 | -1.92 |
| No Eligible (%) | 29.82 % | 27.28 % | -2.54 pp | 54.86 % | +25.04 pp | 24.33 % | -5.49 pp | 35.77 % | +5.95 pp |
| IPC Active | 2.73 inst/cycle | 2.88 inst/cycle | +0.15 | 1.61 inst/cycle | -1.12 | 2.96 inst/cycle | +0.23 | 2.42 inst/cycle | -0.31 |


### Key Learnings

#### ✅ Plan A: `add_evict_last_cache_hint_to_reduce_l1tex_stall` (1.088×)

Adding eviction_policy='evict_last' to all tl.load calls in a 3x3 max_pool2d kernel achieved a modest 1.09x speedup despite the L1/TEX hit rate barely moving (1.41% → 1.5%) and No-Eligible stalls actually worsening (35.6% → 54.9%). The speedup appears to be a confound: the autotune simultaneously added num_warps=1 and num_warps=16 configs, causing autotune to select a 32-thread block (num_warps=1) instead of the prior 64-thread block, which reduced total executed instructions from 2.0M to 1.49M and registers per thread jumped from 32 to 50. The evict_last hint itself did not measurably improve L1 hit rate or reduce L1TEX stall cycles — the CPIStall stall fraction actually increased (50.6% → 59.5%) — suggesting the cache hint had negligible effect on this workload's access pattern.

> **Applicable when**: Kernel has near-zero L1/TEX hit rate (<5%), high L2 hit rate (>70%), and CPIStall rule flags L1TEX scoreboard dependency as >50% of stall cycles. The input tensor has spatial locality that could be exploited by neighboring blocks sharing pooling window rows.

> **Avoid when**: The working set per SM wave is much larger than L1 cache capacity (e.g., >192KB on A100/H200), making cross-block L1 reuse impossible regardless of eviction policy. In such cases, evict_last will not improve L1 hit rate and the CPIStall stall fraction may worsen as the hint competes with other data for limited L1 capacity. Also avoid conflating this change with simultaneous autotune config additions — isolate the cache hint change to measure its true effect.

#### ✅ Plan B: `add_evict_last_cache_hint_to_reduce_l1tex_stall` (1.088×)

Adding eviction_policy='evict_last' to all tl.load calls in a 3x3 max_pool2d kernel achieved a modest 1.09x speedup despite the L1/TEX hit rate barely moving (1.41% → 1.5%) and No-Eligible stalls actually worsening (35.6% → 54.9%). The speedup appears to be a confound: the autotune simultaneously added num_warps=1 and num_warps=16 configs, causing autotune to select a 32-thread block (num_warps=1) instead of the prior 64-thread block, which reduced total executed instructions from 2.0M to 1.49M and registers per thread jumped from 32 to 50. The evict_last hint itself did not measurably improve L1 hit rate or reduce L1TEX stall cycles — the CPIStall stall fraction actually increased (50.6% → 59.5%) — suggesting the cache hint had negligible effect on this workload's access pattern.

> **Applicable when**: Kernel has near-zero L1/TEX hit rate (<5%), high L2 hit rate (>70%), and CPIStall rule flags L1TEX scoreboard dependency as >50% of stall cycles. The input tensor has spatial locality that could be exploited by neighboring blocks sharing pooling window rows.

> **Avoid when**: The working set per SM wave is much larger than L1 cache capacity (e.g., >192KB on A100/H200), making cross-block L1 reuse impossible regardless of eviction policy. In such cases, evict_last will not improve L1 hit rate and the CPIStall stall fraction may worsen as the hint competes with other data for limited L1 capacity. Also avoid conflating this change with simultaneous autotune config additions — isolate the cache hint change to measure its true effect.

#### ✅ Plan C: `add_eviction_policy_hints_to_reduce_l1tex_stalls` (1.088×)

Adding eviction_policy='evict_last' to all input loads and eviction_policy='evict_first' to output stores in a 3x3 max-pool kernel reduced L1TEX scoreboard stalls (CPIStall from 50.6% to 32.4% of CPI) and improved scheduler eligibility (Eligible Warps Per Scheduler: 2.58→4.31, No Eligible: 35.6%→24.3%), yielding a modest 8.8% speedup. However, the L1/TEX hit rate barely moved (1.41%→1.51%), indicating the eviction hints did not meaningfully increase cache reuse — the improvement came primarily from reduced stall cycles per instruction (15.46→13.5 cycles/warp) rather than cache hit rate gains. The elapsed cycles actually increased slightly (11204→11525), suggesting the speedup measured at the end-to-end level may reflect reduced scheduling overhead rather than true memory access improvement.

> **Applicable when**: Kernel has high L1TEX CPIStall (>40% of CPI), very low L1/TEX hit rate (<5%), and performs multiple loads from spatially overlapping neighborhoods (e.g., pooling, convolution, stencil). The stall type is L1TEX scoreboard dependency. Eligible Warps Per Scheduler is low (<3.0) indicating insufficient warp-level latency hiding.

> **Avoid when**: L1/TEX hit rate is already moderate (>10%) or the kernel is purely streaming with no spatial reuse opportunity. Also avoid if instruction count is already high — the after profile shows 25% more executed instructions (2.0M→2.5M), suggesting the eviction policy hints may increase PTX instruction count in some compiler paths. Do not expect large speedups (>20%) from this change alone when block size is small (64 threads / 2 warps), as inter-warp reuse is limited.

---

## Iteration 5

**Best so far**: 1.1206× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_num_stages1_to_reduce_register_pressure | 1.078× | 1.125× | ✓ |  |
| B | add_num_stages_software_pipelining_to_hide_l1tex_latency | 1.071× | 1.076× | ✓ |  |
| C | cap_registers_with_maxnreg_to_improve_sm_occupancy_and_reduce_not_selected_stalls | 1.077× | 1.063× | ✓ |  |
| D | reduce_num_warps_to_1_to_minimize_instruction_count | 1.092× | 1.088× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | add_num_stages_so... | Δ | cap_registers_wit... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8640 ns | 7680 ns | -960 ns (-11.1%) | 8030 ns | -610 ns (-7.1%) | 8130 ns | -510 ns (-5.9%) | 7940 ns | -700 ns (-8.1%) |
| SM Throughput (%) | 55.71 % | 34.28 % | -21.43 pp | 24.91 % | -30.80 pp | 40.22 % | -15.49 pp | 24.24 % | -31.47 pp |
| Memory Throughput (%) | 39.22 % | 39.05 % | -0.17 pp | 33.14 % | -6.08 pp | 38.21 % | -1.01 pp | 33.14 % | -6.08 pp |
| DRAM Throughput (%) | 9.79 % | 11.01 % | +1.22 pp | 10.57 % | +0.78 pp | 10.42 % | +0.63 pp | 10.68 % | +0.89 pp |
| Achieved Occupancy (%) | 77.51 % | 62.88 % | -14.63 pp | 26.68 % | -50.83 pp | 59.36 % | -18.15 pp | 26.93 % | -50.58 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 48.00 register/thread | +26 | 24.00 register/thread | +2 | 50.00 register/thread | +28 |
| Waves/SM | 3.83 | 1.91 | -1.92 | 1.91 | -1.92 | 1.91 | -1.92 | 1.91 | -1.92 |
| No Eligible (%) | 29.82 % | 35.97 % | +6.15 pp | 54.46 % | +24.64 pp | 31.00 % | +1.18 pp | 56.35 % | +26.53 pp |
| IPC Active | 2.73 inst/cycle | 2.50 inst/cycle | -0.23 | 1.74 inst/cycle | -0.99 | 2.73 inst/cycle | +0.00 | 1.75 inst/cycle | -0.98 |


### Key Learnings

#### ✅ Plan B: `add_num_stages_software_pipelining_to_hide_l1tex_latency` (1.076×)

Adding num_stages=2/3 to autotune configs for a 3x3 max-pool kernel caused autotune to select a 32-thread block (num_warps=1, num_stages=1), reducing executed instructions from 2.0M to 1.53M and Warp Cycles Per Issued Instruction from 15.8 to 9.58, but simultaneously collapsing achieved occupancy from 61.9% to 26.7% and increasing No-Eligible stalls from 35.8% to 54.5%. The net result was a marginal 7.6% speedup despite elapsed cycles actually increasing (11416 → 11780). The speedup appears to be driven by the autotune selecting a smaller block size (32 vs 64 threads) rather than by software pipelining itself — the selected config was num_stages=1, not 2 or 3. Adding num_stages variants to autotune configs for pointwise kernels with complex masked loads does not guarantee that the compiler generates async prefetch instructions; the autotune may simply select a lower num_warps config that reduces instruction count at the cost of occupancy.

> **Applicable when**: Kernel has high L1TEX CPIStall (>40% of CPI) and the autotune search space does not include low num_warps configs (num_warps=1). Adding num_stages variants alongside num_warps=1 may allow autotune to discover a smaller block size that reduces total instruction count even if software pipelining itself is not activated.

> **Avoid when**: Kernel already has num_warps=1 in its autotune search space; the dominant bottleneck is occupancy (achieved occupancy already below 30%); the kernel uses complex masked loads with compound boolean predicates where Triton's compiler is unlikely to generate async prefetch instructions for num_stages>1. Do not expect num_stages>1 to activate software pipelining for pointwise kernels with 9+ masked conditional loads — the compiler typically only pipelines simple, unconditional loads.

---

## Cumulative Progress

```
Baseline:  8640 ns  (1.000×)
Iter 2     7710 ns  (1.121×)  — fix_autotune_grid_lambda_and_add_low_warp_configs

Final best: 7940 ns  (1.0882×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=1),  # NEW primary: 32 threads/block
+        triton.Config({'XBLOCK': 128}, num_warps=2),  # keep as fallback
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1034880
@@ -25,57 +33,64 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-7920) + x6), tmp10 & xmask, other=float("-inf"))
+    tmp10_masked = tmp10 & xmask
+    tmp11 = tl.load(in_ptr0 + ((-7920) + x6), mask=tmp10_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp12 = x1
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-7392) + x6), tmp16 & xmask, other=float("-inf"))
+    tmp16_masked = tmp16 & xmask
+    tmp17 = tl.load(in_ptr0 + ((-7392) + x6), mask=tmp16_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + x1
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-6864) + x6), tmp23 & xmask, other=float("-inf"))
+    tmp23_masked = tmp23 & xmask
+    tmp24 = tl.load(in_ptr0 + ((-6864) + x6), mask=tmp23_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = x2
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-528) + x6), tmp30 & xmask, other=float("-inf"))
+    tmp30_masked = tmp30 & xmask
+    tmp31 = tl.load(in_ptr0 + ((-528) + x6), mask=tmp30_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=float("-inf"))
+    tmp33_masked = tmp33 & xmask
+    tmp34 = tl.load(in_ptr0 + (x6), mask=tmp33_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (528 + x6), tmp36 & xmask, other=float("-inf"))
+    tmp36_masked = tmp36 & xmask
+    tmp37 = tl.load(in_ptr0 + (528 + x6), mask=tmp36_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + x2
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (6864 + x6), tmp43 & xmask, other=float("-inf"))
+    tmp43_masked = tmp43 & xmask
+    tmp44 = tl.load(in_ptr0 + (6864 + x6), mask=tmp43_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (7392 + x6), tmp46 & xmask, other=float("-inf"))
+    tmp46_masked = tmp46 & xmask
+    tmp47 = tl.load(in_ptr0 + (7392 + x6), mask=tmp46_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (7920 + x6), tmp49 & xmask, other=float("-inf"))
+    tmp49_masked = tmp49 & xmask
+    tmp50 = tl.load(in_ptr0 + (7920 + x6), mask=tmp49_masked, other=float("-inf"), eviction_policy='evict_last')
     tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (x6), tmp51, xmask)
-
-
+    tl.store(out_ptr0 + (x6), tmp51, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 1034880
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_64[grid](arg_0, arg_1, 1034880, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_64[grid](arg_0, arg_1, 1034880)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
