# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3040 ns — **Best latency**: 2690 ns (**1.1301×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_better_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (640,1,1), Waves/SM: 0.30
- **Registers/thread**: 28
- **Duration**: 3040 ns
- **SM Throughput**: 12.12 % | **Memory**: 11.17 % | **DRAM**: 1.28 %
- **Achieved Occupancy**: 26.34 %
- **IPC Active**: 1.02 inst/cycle | **No Eligible**: 68.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_better_occupancy | — | 1.130× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3040 ns | 2690 ns | -350 ns (-11.5%) |
| SM Throughput (%) | 12.12 % | 2.42 % | -9.70 pp |
| Memory Throughput (%) | 11.17 % | 1.43 % | -9.74 pp |
| DRAM Throughput (%) | 1.28 % | 1.43 % | +0.15 pp |
| Achieved Occupancy (%) | 26.34 % | 9.34 % | -17.00 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.30 | 0.15 | -0.15 |
| No Eligible (%) | 68.91 % | 87.15 % | +18.24 pp |
| IPC Active | 1.02 inst/cycle | 0.24 inst/cycle | -0.78 |

---

## Cumulative Progress

```
Baseline:  3040 ns  (1.000×)
Iter 1     2690 ns  (1.130×)  — autotune_tile_sizes_for_better_occupancy

Final best: 2690 ns  (1.1301×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 1, 'YBLOCK': 32},  num_warps=2),
+    triton.Config({'XBLOCK': 1, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8),
+    triton.Config({'XBLOCK': 2, 'YBLOCK': 32},  num_warps=2),
+    triton.Config({'XBLOCK': 2, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 2, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 4, 'YBLOCK': 32},  num_warps=2),
+    triton.Config({'XBLOCK': 4, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 8, 'YBLOCK': 32},  num_warps=4),
+    triton.Config({'XBLOCK': 8, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 5120
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +36,8 @@

     y3 = yindex
     y0 = (yindex % 160)
     y1 = yindex // 160
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

     ynumel = 5120
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, 5120, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9[grid](arg_0, arg_1, 5120, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
