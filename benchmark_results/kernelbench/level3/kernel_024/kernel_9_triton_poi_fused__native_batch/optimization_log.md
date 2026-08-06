# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2270 ns — **Best latency**: 2270 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 16
- **Duration**: 2270 ns
- **SM Throughput**: 0.01 % | **Memory**: 0.40 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 5.27 %
- **IPC Active**: 0.12 inst/cycle | **No Eligible**: 96.38 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 0.963× | 0.934× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2270 ns | 2430 ns | +160 ns (+7.0%) |
| SM Throughput (%) | 0.01 % | 0.02 % | +0.01 pp |
| Memory Throughput (%) | 0.40 % | 0.39 % | -0.01 pp |
| DRAM Throughput (%) | 0.04 % | 0.03 % | -0.01 pp |
| Achieved Occupancy (%) | 5.27 % | 5.30 % | +0.03 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 96.38 % | 96.02 % | -0.36 pp |
| IPC Active | 0.12 inst/cycle | 0.12 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_single_cta_for_loop_tile_iteration | 0.994× | 0.970× | ✓ |  |
| B | collapse_to_single_cta_with_xblock_256 | 0.997× | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_single... | Δ | collapse_to_singl... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2270 ns | 2340 ns | +70 ns (+3.1%) | 2210 ns | -60 ns (-2.6%) |
| SM Throughput (%) | 0.01 % | 0.01 % | +0.00 pp | 0.01 % | +0.00 pp |
| Memory Throughput (%) | 0.40 % | 0.43 % | +0.03 pp | 0.42 % | +0.02 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 5.27 % | 1.56 % | -3.71 pp | 5.27 % | +0.00 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 96.38 % | 91.27 % | -5.11 pp | 97.04 % | +0.66 pp |
| IPC Active | 0.12 inst/cycle | 0.07 inst/cycle | -0.05 | 0.12 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  2270 ns  (1.000×)

Final best: 2270 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 192
@@ -16,13 +27,11 @@

     tmp1 = tl.sigmoid(tmp0)
     tl.store(in_out_ptr0 + (x0), tmp1, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     xnumel = 192
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9[grid](arg_0, 192, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_9[grid](arg_0, 192)
     return arg_0
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
