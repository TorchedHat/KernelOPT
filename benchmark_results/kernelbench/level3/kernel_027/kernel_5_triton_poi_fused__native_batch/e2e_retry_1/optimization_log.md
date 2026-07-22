# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3100 ns — **Best latency**: 3300 ns (**0.9394×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 28
- **Duration**: 3100 ns
- **SM Throughput**: 16.37 % | **Memory**: 15.54 % | **DRAM**: 1.97 %
- **Achieved Occupancy**: 34.90 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 58.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 1.034× | 0.939× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3100 ns | 3300 ns | +200 ns (+6.5%) |
| SM Throughput (%) | 16.37 % | 17.22 % | +0.85 pp |
| Memory Throughput (%) | 15.54 % | 16.33 % | +0.79 pp |
| DRAM Throughput (%) | 1.97 % | 1.87 % | -0.10 pp |
| Achieved Occupancy (%) | 34.90 % | 36.89 % | +1.99 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.48 | 0.48 | +0.00 |
| No Eligible (%) | 58.31 % | 58.96 % | +0.65 pp |
| IPC Active | 1.55 inst/cycle | 1.61 inst/cycle | +0.06 |

---

## Cumulative Progress

```
Baseline:  3100 ns  (1.000×)

Final best: 3300 ns  (0.9394×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+], key=['xnumel', 'ynumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
@@ -19,10 +26,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +35,7 @@

     ynumel = 8192
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5[grid](arg_0, arg_1, 8192, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5[grid](arg_0, arg_1, 8192, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
