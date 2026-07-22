# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3070 ns — **Best latency**: 2780 ns (**1.1043×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy`** beats baseline latency.

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
- **Duration**: 3070 ns
- **SM Throughput**: 15.25 % | **Memory**: 13.82 % | **DRAM**: 1.75 %
- **Achieved Occupancy**: 33.16 %
- **IPC Active**: 1.60 inst/cycle | **No Eligible**: 55.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_full_sm_utilization | 1.004× | 1.129× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy | 1.010× | 1.104× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2720 ns | -350 ns (-11.4%) | 2780 ns | -290 ns (-9.4%) |
| SM Throughput (%) | 15.25 % | 15.60 % | +0.35 pp | 10.57 % | -4.68 pp |
| Memory Throughput (%) | 13.82 % | 7.01 % | -6.81 pp | 8.97 % | -4.85 pp |
| DRAM Throughput (%) | 1.75 % | 1.99 % | +0.24 pp | 1.92 % | +0.17 pp |
| Achieved Occupancy (%) | 33.16 % | 33.81 % | +0.65 pp | 15.06 % | -18.10 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 30.00 register/thread | +2 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.21 | -0.21 |
| No Eligible (%) | 55.63 % | 51.69 % | -3.94 pp | 62.69 % | +7.06 pp |
| IPC Active | 1.60 inst/cycle | 1.95 inst/cycle | +0.35 | 1.33 inst/cycle | -0.27 |

---

## Cumulative Progress

```
Baseline:  3070 ns  (1.000×)
Iter 1     2780 ns  (1.104×)  — autotune_tile_sizes_for_occupancy

Final best: 2780 ns  (1.1043×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 256}, num_warps=8),
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
@@ -19,10 +33,8 @@

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
@@ -30,7 +42,7 @@

     ynumel = 7168
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
