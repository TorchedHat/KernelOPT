# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3140 ns — **Best latency**: 2820 ns (**1.1135×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_persistent_scheduling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_14_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (896,1,1), Waves/SM: 0.42
- **Registers/thread**: 28
- **Duration**: 3140 ns
- **SM Throughput**: 16.00 % | **Memory**: 14.47 % | **DRAM**: 1.72 %
- **Achieved Occupancy**: 34.96 %
- **IPC Active**: 1.65 inst/cycle | **No Eligible**: 57.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_occupancy_and_persistent_scheduling | 1.025× | 1.113× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3140 ns | 2820 ns | -320 ns (-10.2%) |
| SM Throughput (%) | 16.00 % | 9.99 % | -6.01 pp |
| Memory Throughput (%) | 14.47 % | 8.47 % | -6.00 pp |
| DRAM Throughput (%) | 1.72 % | 1.92 % | +0.20 pp |
| Achieved Occupancy (%) | 34.96 % | 14.21 % | -20.75 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 |
| Waves/SM | 0.42 | 0.21 | -0.21 |
| No Eligible (%) | 57.78 % | 62.89 % | +5.11 pp |
| IPC Active | 1.65 inst/cycle | 1.24 inst/cycle | -0.41 |

---

## Cumulative Progress

```
Baseline:  3140 ns  (1.000×)
Iter 1     2820 ns  (1.113×)  — autotune_tile_sizes_for_occupancy_and_persistent_scheduling

Final best: 2820 ns  (1.1135×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 256}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 7168
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 224)
     y1 = yindex // 224
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 224*x2 + 2016*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 224*x2 + 2016*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 7168
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
