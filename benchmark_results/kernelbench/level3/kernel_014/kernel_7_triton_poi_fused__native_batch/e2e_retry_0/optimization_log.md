# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2750 ns — **Best latency**: 2850 ns (**0.9649×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 28
- **Duration**: 2750 ns
- **SM Throughput**: 9.32 % | **Memory**: 8.79 % | **DRAM**: 1.13 %
- **Achieved Occupancy**: 21.91 %
- **IPC Active**: 0.89 inst/cycle | **No Eligible**: 70.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_underutilized_small_grid | 1.044× | 1.034× | ✓ |  |
| B | retile_and_autotune_permute_kernel | 1.034× | 1.100× | ✓ |  |
| C | add_triton_autotune | 1.047× | 0.965× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | retile_and_autotu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2750 ns | 2660 ns | -90 ns (-3.3%) | 2500 ns | -250 ns (-9.1%) | 2850 ns | +100 ns (+3.6%) |
| SM Throughput (%) | 9.32 % | 5.93 % | -3.39 pp | 1.57 % | -7.75 pp | 9.25 % | -0.07 pp |
| Memory Throughput (%) | 8.79 % | 5.47 % | -3.32 pp | 1.25 % | -7.54 pp | 8.73 % | -0.06 pp |
| DRAM Throughput (%) | 1.13 % | 1.18 % | +0.05 pp | 1.25 % | +0.12 pp | 1.11 % | -0.02 pp |
| Achieved Occupancy (%) | 21.91 % | 8.31 % | -13.60 pp | 8.39 % | -13.52 pp | 20.45 % | -1.46 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.24 | +0.00 |
| No Eligible (%) | 70.20 % | 75.16 % | +4.96 pp | 89.75 % | +19.55 pp | 72.33 % | +2.13 pp |
| IPC Active | 0.89 inst/cycle | 0.74 inst/cycle | -0.15 | 0.20 inst/cycle | -0.69 | 1.06 inst/cycle | +0.17 |

---

## Cumulative Progress

```
Baseline:  2750 ns  (1.000×)

Final best: 2850 ns  (0.9649×)
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
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 8}, num_warps=8, num_stages=1),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
@@ -19,10 +26,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +35,7 @@

     ynumel = 4096
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7[grid](arg_0, arg_1, 4096, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7[grid](arg_0, arg_1, 4096, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
