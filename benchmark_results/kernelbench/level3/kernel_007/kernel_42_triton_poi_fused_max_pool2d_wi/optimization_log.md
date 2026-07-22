# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_42` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8540 ns — **Best latency**: 7520 ns (**1.1356×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_tune_num_warps_for_alu_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_42`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_42_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 22
- **Duration**: 8540 ns
- **SM Throughput**: 56.28 % | **Memory**: 27.09 % | **DRAM**: 9.62 %
- **Achieved Occupancy**: 73.20 %
- **IPC Active**: 2.71 inst/cycle | **No Eligible**: 31.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_eliminate_partial_wave | 1.001× | 0.995× | ✓ |  |
| C | autotune_xblock_vectorized_loads | 1.004× | 0.478× | ✓ | **✓ best** |
| D | increase_xblock_to_eliminate_partial_wave_tail | 0.858× | 0.839× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_e... | Δ | autotune_xblock_v... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8540 ns | 8580 ns | +40 ns (+0.5%) | 17860 ns | +9320 ns (+109.1%) | 10180 ns | +1640 ns (+19.2%) |
| SM Throughput (%) | 56.28 % | 56.16 % | -0.12 pp | 73.78 % | +17.50 pp | 64.62 % | +8.34 pp |
| Memory Throughput (%) | 27.09 % | 26.87 % | -0.22 pp | 26.01 % | -1.08 pp | 29.75 % | +2.66 pp |
| DRAM Throughput (%) | 9.62 % | 9.59 % | -0.03 pp | 4.59 % | -5.03 pp | 8.06 % | -1.56 pp |
| Achieved Occupancy (%) | 73.20 % | 76.52 % | +3.32 pp | 86.22 % | +13.02 pp | 80.37 % | +7.17 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 7.42 | +3.71 | 3.71 | +0.00 |
| No Eligible (%) | 31.00 % | 30.25 % | -0.75 pp | 19.64 % | -11.36 pp | 20.72 % | -10.28 pp |
| IPC Active | 2.71 inst/cycle | 2.72 inst/cycle | +0.01 | 3.20 inst/cycle | +0.49 | 3.13 inst/cycle | +0.42 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_vectorized_loads` (0.478×)

Adding @triton.autotune with larger XBLOCK configs (256, 512, 1024) to a Triton pointwise kernel caused a 2x slowdown despite improving achieved occupancy from 73% to 86%. The autotune selected XBLOCK=256 (block_size doubled), which increased registers per thread from 22 to 32, reducing blocks-per-SM from 21 to 8 (Block Limit Registers), and caused total executed instructions to balloon from 3.4M to 9.9M — nearly 3x more work per element. The kernel duration increased from 8.54 µs to 17.86 µs. The root cause is that wider XBLOCK in a kernel with many conditional loads (9 masked tl.load calls with complex index arithmetic) dramatically increases register pressure and instruction count without proportional memory bandwidth gains — memory throughput stayed flat at ~27% while compute became the bottleneck at 74% SM busy.

> **Applicable when**: Never applicable as described — autotune with larger XBLOCK is harmful for kernels with many conditional masked loads and high per-element instruction count.

> **Avoid when**: Kernel contains many (≥5) conditional masked loads with complex index arithmetic per output element; register pressure is already moderate (22+ regs/thread at baseline); the kernel is compute-bound or instruction-count-bound rather than memory-bandwidth-bound; memory throughput SOL does not improve with wider tiles. Specifically: when executed instruction count grows super-linearly with XBLOCK, do not increase XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_increase_num_warps_for_latency_hiding | 1.021× | 1.011× | ✓ |  |
| B | fix_autotune_implementation_flaw_revert_to_fixed_xblock | 1.001× | 1.007× | ✓ |  |
| C | revert_to_xblock128_remove_broken_autotune | 1.021× | 1.026× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_impl... | Δ | revert_to_xblock1... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8540 ns | 8450 ns | -90 ns (-1.1%) | 8480 ns | -60 ns (-0.7%) | 8320 ns | -220 ns (-2.6%) |
| SM Throughput (%) | 56.28 % | 55.70 % | -0.58 pp | 56.17 % | -0.11 pp | 55.74 % | -0.54 pp |
| Memory Throughput (%) | 27.09 % | 27.58 % | +0.49 pp | 27.46 % | +0.37 pp | 27.91 % | +0.82 pp |
| DRAM Throughput (%) | 9.62 % | 9.72 % | +0.10 pp | 9.70 % | +0.08 pp | 9.90 % | +0.28 pp |
| Achieved Occupancy (%) | 73.20 % | 78.38 % | +5.18 pp | 77.65 % | +4.45 pp | 77.16 % | +3.96 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 31.00 % | 31.71 % | +0.71 pp | 30.45 % | -0.55 pp | 29.88 % | -1.12 pp |
| IPC Active | 2.71 inst/cycle | 2.77 inst/cycle | +0.06 | 2.73 inst/cycle | +0.02 | 2.75 inst/cycle | +0.04 |

---

## Iteration 3

**Best so far**: 1.0264× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_alleviate_scheduler_contention_and_improve_l1_locality | 1.000× | 1.011× | ✓ |  |
| B | fix_grid_lambda_and_tune_num_warps_for_alu_latency_hiding | 1.095× | 1.136× | ✓ | **✓ best** |
| C | increase_num_warps_for_latency_hiding_fixed_xblock | 1.000× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | fix_grid_lambda_a... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8540 ns | 8450 ns | -90 ns (-1.1%) | 7520 ns | -1020 ns (-11.9%) | 8480 ns | -60 ns (-0.7%) |
| SM Throughput (%) | 56.28 % | 55.24 % | -1.04 pp | 35.35 % | -20.93 pp | 55.66 % | -0.62 pp |
| Memory Throughput (%) | 27.09 % | 27.17 % | +0.08 pp | 30.70 % | +3.61 pp | 27.37 % | +0.28 pp |
| DRAM Throughput (%) | 9.62 % | 9.74 % | +0.12 pp | 10.93 % | +1.31 pp | 9.69 % | +0.07 pp |
| Achieved Occupancy (%) | 73.20 % | 77.00 % | +3.80 pp | 57.45 % | -15.75 pp | 78.52 % | +5.32 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 1.86 | -1.85 | 3.71 | +0.00 |
| No Eligible (%) | 31.00 % | 30.06 % | -0.94 pp | 38.64 % | +7.64 pp | 31.99 % | +0.99 pp |
| IPC Active | 2.71 inst/cycle | 2.75 inst/cycle | +0.04 | 2.54 inst/cycle | -0.17 | 2.65 inst/cycle | -0.06 |

---

## Iteration 4

**Best so far**: 1.1356× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_sm_count_eliminate_partial_wave_via_xblock_256_num_warps_1 | 1.073× | 1.004× | ✓ |  |
| B | autotune_num_warps_fixed_xblock_reduce_instruction_count | 1.074× | 1.131× | ✓ |  |
| C | add_cache_eviction_hints_to_reduce_l2_pressure | 0.958× | 0.923× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_grid_to_sm_... | Δ | autotune_num_warp... | Δ | add_cache_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8540 ns | 8510 ns | -30 ns (-0.4%) | 7550 ns | -990 ns (-11.6%) | 9250 ns | +710 ns (+8.3%) |
| SM Throughput (%) | 56.28 % | 30.23 % | -26.05 pp | 35.52 % | -20.76 pp | 58.61 % | +2.33 pp |
| Memory Throughput (%) | 27.09 % | 24.40 % | -2.69 pp | 30.20 % | +3.11 pp | 24.92 % | -2.17 pp |
| DRAM Throughput (%) | 9.62 % | 9.67 % | +0.05 pp | 10.89 % | +1.27 pp | 8.88 % | -0.74 pp |
| Achieved Occupancy (%) | 73.20 % | 29.92 % | -43.28 pp | 57.28 % | -15.92 pp | 79.06 % | +5.86 pp |
| Registers/Thread | 22.00 register/thread | 64.00 register/thread | +42 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 1.86 | -1.85 | 3.71 | +0.00 |
| No Eligible (%) | 31.00 % | 45.19 % | +14.19 pp | 34.61 % | +3.61 pp | 23.08 % | -7.92 pp |
| IPC Active | 2.71 inst/cycle | 2.08 inst/cycle | -0.63 | 2.53 inst/cycle | -0.18 | 3.02 inst/cycle | +0.31 |

---

## Iteration 5

**Best so far**: 1.1356× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipelining_to_hide_no_eligible_stalls | 1.090× | 1.098× | ✓ |  |
| B | fix_grid_mismatch_and_sweep_num_warps_for_l1tex_latency_hiding | 1.003× | 0.985× | ✓ |  |
| C | add_num_stages_software_pipelining_to_hide_load_latency | 0.937× | 0.917× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | fix_grid_mismatch... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8540 ns | 7780 ns | -760 ns (-8.9%) | 8670 ns | +130 ns (+1.5%) | 9310 ns | +770 ns (+9.0%) |
| SM Throughput (%) | 56.28 % | 21.41 % | -34.87 pp | 56.00 % | -0.28 pp | 59.29 % | +3.01 pp |
| Memory Throughput (%) | 27.09 % | 26.68 % | -0.41 pp | 26.29 % | -0.80 pp | 24.57 % | -2.52 pp |
| DRAM Throughput (%) | 9.62 % | 10.58 % | +0.96 pp | 9.46 % | -0.16 pp | 8.81 % | -0.81 pp |
| Achieved Occupancy (%) | 73.20 % | 23.25 % | -49.95 pp | 77.89 % | +4.69 pp | 78.44 % | +5.24 pp |
| Registers/Thread | 22.00 register/thread | 50.00 register/thread | +28 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 31.00 % | 59.24 % | +28.24 pp | 30.31 % | -0.69 pp | 23.14 % | -7.86 pp |
| IPC Active | 2.71 inst/cycle | 1.56 inst/cycle | -1.15 | 2.77 inst/cycle | +0.06 | 3.04 inst/cycle | +0.33 |

---

## Cumulative Progress

```
Baseline:  8540 ns  (1.000×)
Iter 2     8320 ns  (1.026×)  — revert_to_xblock128_remove_broken_autotune
Iter 3     7520 ns  (1.136×)  — fix_grid_lambda_and_tune_num_warps_for_alu_latency_hiding

Final best: 7520 ns  (1.1356×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1003520
@@ -68,13 +77,12 @@

     tmp51 = tl.maximum(tmp48, tmp50)
     tl.store(out_ptr0 + (x6), tmp51, mask=xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 1003520
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_42[grid](arg_0, arg_1, 1003520, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_42[grid](arg_0, arg_1, 1003520)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
