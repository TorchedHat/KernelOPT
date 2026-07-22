# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7260 ns — **Best latency**: 7260 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_26_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7350,1,1), Waves/SM: 3.48
- **Registers/thread**: 18
- **Duration**: 7260 ns
- **SM Throughput**: 28.41 % | **Memory**: 17.11 % | **DRAM**: 10.63 %
- **Achieved Occupancy**: 48.78 %
- **IPC Active**: 1.75 inst/cycle | **No Eligible**: 56.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.940× | 0.938× | ✓ |  |
| B | autotune_xblock_warps_to_improve_occupancy_and_hide_l1tex_latency | 0.964× | 0.986× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_broadcast_reuse | 0.930× | 0.969× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_latency_hiding | 0.962× | 0.973× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_w... | Δ | increase_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7260 ns | 7740 ns | +480 ns (+6.6%) | 7360 ns | +100 ns (+1.4%) | 7490 ns | +230 ns (+3.2%) | 7460 ns | +200 ns (+2.8%) |
| SM Throughput (%) | 28.41 % | 36.95 % | +8.54 pp | 30.50 % | +2.09 pp | 37.16 % | +8.75 pp | 36.96 % | +8.55 pp |
| Memory Throughput (%) | 17.11 % | 23.63 % | +6.52 pp | 21.82 % | +4.71 pp | 23.80 % | +6.69 pp | 23.89 % | +6.78 pp |
| DRAM Throughput (%) | 10.63 % | 9.95 % | -0.68 pp | 10.49 % | -0.14 pp | 10.31 % | -0.32 pp | 10.33 % | -0.30 pp |
| Achieved Occupancy (%) | 48.78 % | 45.00 % | -3.78 pp | 44.13 % | -4.65 pp | 45.17 % | -3.61 pp | 46.34 % | -2.44 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 56.80 % | 40.30 % | -16.50 pp | 47.25 % | -9.55 pp | 40.48 % | -16.32 pp | 39.05 % | -17.75 pp |
| IPC Active | 1.75 inst/cycle | 2.37 inst/cycle | +0.62 | 2.14 inst/cycle | +0.39 | 2.35 inst/cycle | +0.60 | 2.36 inst/cycle | +0.61 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipelining_to_hide_l1tex_latency | 0.990× | 1.004× | ✓ |  |
| B | fix_autotune_grid_and_add_pipeline_stages_for_latency_hiding | 0.939× | 0.969× | ✓ |  |
| C | fix_grid_calculation_and_add_num_stages_for_latency_hiding | 0.962× | 0.969× | ✓ |  |
| D | fix_hardcoded_grid_to_enable_autotune_xblock_scaling | 0.938× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | fix_autotune_grid... | Δ | fix_grid_calculat... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7260 ns | 7230 ns | -30 ns (-0.4%) | 7490 ns | +230 ns (+3.2%) | 7490 ns | +230 ns (+3.2%) | 7490 ns | +230 ns (+3.2%) |
| SM Throughput (%) | 28.41 % | 31.92 % | +3.51 pp | 37.21 % | +8.80 pp | 37.00 % | +8.59 pp | 36.79 % | +8.38 pp |
| Memory Throughput (%) | 17.11 % | 23.01 % | +5.90 pp | 24.07 % | +6.96 pp | 23.75 % | +6.64 pp | 23.71 % | +6.60 pp |
| DRAM Throughput (%) | 10.63 % | 10.71 % | +0.08 pp | 10.34 % | -0.29 pp | 10.32 % | -0.31 pp | 10.30 % | -0.33 pp |
| Achieved Occupancy (%) | 48.78 % | 44.61 % | -4.17 pp | 46.10 % | -2.68 pp | 53.19 % | +4.41 pp | 51.22 % | +2.44 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 56.80 % | 49.06 % | -7.74 pp | 39.28 % | -17.52 pp | 40.19 % | -16.61 pp | 40.09 % | -16.71 pp |
| IPC Active | 1.75 inst/cycle | 2.08 inst/cycle | +0.33 | 2.37 inst/cycle | +0.62 | 2.33 inst/cycle | +0.58 | 2.34 inst/cycle | +0.59 |

---

## Cumulative Progress

```
Baseline:  7260 ns  (1.000×)

Final best: 7260 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=8,  num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
@@ -26,7 +38,6 @@

     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + xindex, tmp19, xmask)   # COALESCED: stride-1 store
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -35,7 +46,7 @@

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
