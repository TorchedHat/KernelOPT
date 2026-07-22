# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 7550 ns (**0.9921×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_3_triton_poi_fused_addmm_relu_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8192,1,1), Waves/SM: 3.88
- **Registers/thread**: 16
- **Duration**: 7490 ns
- **SM Throughput**: 21.26 % | **Memory**: 15.13 % | **DRAM**: 11.46 %
- **Achieved Occupancy**: 43.85 %
- **IPC Active**: 1.24 inst/cycle | **No Eligible**: 67.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 0.996× | 0.633× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 1.014× | 0.992× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_reduced_partial_wave | 0.993× | 0.992× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 0.976× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 11840 ns | +4350 ns (+58.1%) | 7550 ns | +60 ns (+0.8%) | 7550 ns | +60 ns (+0.8%) | 7580 ns | +90 ns (+1.2%) |
| SM Throughput (%) | 21.26 % | 61.97 % | +40.71 pp | 25.22 % | +3.96 pp | 24.62 % | +3.36 pp | 21.73 % | +0.47 pp |
| Memory Throughput (%) | 15.13 % | 21.61 % | +6.48 pp | 14.42 % | -0.71 pp | 14.53 % | -0.60 pp | 14.97 % | -0.16 pp |
| DRAM Throughput (%) | 11.46 % | 7.24 % | -4.22 pp | 11.41 % | -0.05 pp | 11.39 % | -0.07 pp | 11.31 % | -0.15 pp |
| Achieved Occupancy (%) | 43.85 % | 69.75 % | +25.90 pp | 37.22 % | -6.63 pp | 38.56 % | -5.29 pp | 43.43 % | -0.42 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3.88 | 15.52 | +11.64 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 67.67 % | 24.51 % | -43.16 pp | 58.23 % | -9.44 pp | 59.93 % | -7.74 pp | 67.16 % | -0.51 pp |
| IPC Active | 1.24 inst/cycle | 2.99 inst/cycle | +1.75 | 1.67 inst/cycle | +0.43 | 1.61 inst/cycle | +0.37 | 1.28 inst/cycle | +0.04 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_autotune_num_warps_to_hide_l1tex_latency | 0.967× | 0.972× | ✓ |  |
| B | fix_grid_lambda_to_match_autotuned_xblock_and_increase_num_warps | 1.000× | 1.004× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_correct_xblock_tuning | 0.998× | 1.004× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_for_correct_autotune_xblock_scaling | 1.009× | 0.633× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7710 ns | +220 ns (+2.9%) | 7460 ns | -30 ns (-0.4%) | 7460 ns | -30 ns (-0.4%) | 11840 ns | +4350 ns (+58.1%) |
| SM Throughput (%) | 21.26 % | 24.27 % | +3.01 pp | 21.27 % | +0.01 pp | 21.70 % | +0.44 pp | 63.08 % | +41.82 pp |
| Memory Throughput (%) | 15.13 % | 14.03 % | -1.10 pp | 15.14 % | +0.01 pp | 15.25 % | +0.12 pp | 21.87 % | +6.74 pp |
| DRAM Throughput (%) | 11.46 % | 11.16 % | -0.30 pp | 11.51 % | +0.05 pp | 11.51 % | +0.05 pp | 7.24 % | -4.22 pp |
| Achieved Occupancy (%) | 43.85 % | 38.78 % | -5.07 pp | 44.16 % | +0.31 pp | 43.08 % | -0.77 pp | 70.11 % | +26.26 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 | 15.52 | +11.64 |
| No Eligible (%) | 67.67 % | 56.38 % | -11.29 pp | 67.62 % | -0.05 pp | 67.03 % | -0.64 pp | 24.52 % | -43.15 pp |
| IPC Active | 1.24 inst/cycle | 1.64 inst/cycle | +0.40 | 1.23 inst/cycle | -0.01 | 1.30 inst/cycle | +0.06 | 2.97 inst/cycle | +1.73 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_unlocking_autotune_search | 0.996× | 1.004× | ✓ |  |
| B | fix_grid_lambda_to_correctly_scale_with_autotuned_xblock | 0.979× | 1.004× | ✓ |  |
| C | fix_grid_lambda_and_add_num_stages_to_hide_l1tex_latency | 0.967× | 0.992× | ✓ |  |
| D | fix_grid_lambda_to_use_meta_xblock_for_correct_autotune_scaling | 0.995× | 0.631× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7460 ns | -30 ns (-0.4%) | 7460 ns | -30 ns (-0.4%) | 7550 ns | +60 ns (+0.8%) | 11870 ns | +4380 ns (+58.5%) |
| SM Throughput (%) | 21.26 % | 21.11 % | -0.15 pp | 21.81 % | +0.55 pp | 25.29 % | +4.03 pp | 63.15 % | +41.89 pp |
| Memory Throughput (%) | 15.13 % | 15.13 % | +0.00 pp | 15.28 % | +0.15 pp | 14.47 % | -0.66 pp | 21.97 % | +6.84 pp |
| DRAM Throughput (%) | 11.46 % | 11.51 % | +0.05 pp | 11.51 % | +0.05 pp | 11.39 % | -0.07 pp | 7.23 % | -4.23 pp |
| Achieved Occupancy (%) | 43.85 % | 42.92 % | -0.93 pp | 42.40 % | -1.45 pp | 37.36 % | -6.49 pp | 69.75 % | +25.90 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 | 15.52 | +11.64 |
| No Eligible (%) | 67.67 % | 68.47 % | +0.80 pp | 68.25 % | +0.58 pp | 59.14 % | -8.53 pp | 24.52 % | -43.15 pp |
| IPC Active | 1.24 inst/cycle | 1.25 inst/cycle | +0.01 | 1.27 inst/cycle | +0.03 | 1.66 inst/cycle | +0.42 | 2.99 inst/cycle | +1.75 |

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)

Final best: 7550 ns  (0.9921×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
+    triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
+    triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=2),
+    triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+    triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=2),
+    triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+    triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+    triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused_addmm_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1048576
@@ -20,13 +31,12 @@

     tmp4 = tl.maximum(tmp3, tmp2)
     tl.store(in_out_ptr0 + (x2), tmp4, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 1048576
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_relu_3[grid](arg_0, arg_1, 1048576, XBLOCK=128)
+    triton_poi_fused_addmm_relu_3[grid](arg_0, arg_1, 1048576)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
