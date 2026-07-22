# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3200 ns — **Best latency**: 3490 ns (**0.9169×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 0.46
- **Registers/thread**: 18
- **Duration**: 3200 ns
- **SM Throughput**: 13.05 % | **Memory**: 4.80 % | **DRAM**: 3.26 %
- **Achieved Occupancy**: 38.77 %
- **IPC Active**: 1.34 inst/cycle | **No Eligible**: 64.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.022× | 0.936× | ✓ |  |
| B | autotune_xblock_and_warps_for_underutilized_pointwise_kernel | 1.036× | 0.917× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.003× | 0.970× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 3420 ns | +220 ns (+6.9%) | 3490 ns | +290 ns (+9.1%) | 3300 ns | +100 ns (+3.1%) |
| SM Throughput (%) | 13.05 % | 10.57 % | -2.48 pp | 9.10 % | -3.95 pp | 12.86 % | -0.19 pp |
| Memory Throughput (%) | 4.80 % | 4.25 % | -0.55 pp | 4.46 % | -0.34 pp | 4.60 % | -0.20 pp |
| DRAM Throughput (%) | 3.26 % | 3.08 % | -0.18 pp | 2.98 % | -0.28 pp | 3.16 % | -0.10 pp |
| Achieved Occupancy (%) | 38.77 % | 27.50 % | -11.27 pp | 20.50 % | -18.27 pp | 40.79 % | +2.02 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 18.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.23 | -0.23 | 0.46 | +0.00 |
| No Eligible (%) | 64.53 % | 71.48 % | +6.95 pp | 73.52 % | +8.99 pp | 63.89 % | -0.64 pp |
| IPC Active | 1.34 inst/cycle | 0.96 inst/cycle | -0.38 | 0.87 inst/cycle | -0.47 | 1.15 inst/cycle | -0.19 |

---

## Cumulative Progress

```
Baseline:  3200 ns  (1.000×)

Final best: 3490 ns  (0.9169×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 125440
@@ -31,8 +45,6 @@

     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -41,7 +53,7 @@

     arg_4 = inputs[4]
     xnumel = 125440
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 125440, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 125440)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
