# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7330 ns — **Best latency**: 6530 ns (**1.1225×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_eliminate_partial_wave_and_thread_waste`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_225_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7168,1,1), Waves/SM: 3.39
- **Registers/thread**: 28
- **Duration**: 7330 ns
- **SM Throughput**: 54.09 % | **Memory**: 51.11 % | **DRAM**: 5.79 %
- **Achieved Occupancy**: 71.88 %
- **IPC Active**: 3.14 inst/cycle | **No Eligible**: 21.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_eliminate_partial_wave_and_thread_waste | 1.033× | 1.123× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.022× | 1.041× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_el... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7330 ns | 6530 ns | -800 ns (-10.9%) | 7040 ns | -290 ns (-4.0%) |
| SM Throughput (%) | 54.09 % | 19.39 % | -34.70 pp | 52.57 % | -1.52 pp |
| Memory Throughput (%) | 51.11 % | 6.49 % | -44.62 pp | 49.65 % | -1.46 pp |
| DRAM Throughput (%) | 5.79 % | 6.49 % | +0.70 pp | 6.00 % | +0.21 pp |
| Achieved Occupancy (%) | 71.88 % | 23.89 % | -47.99 pp | 71.50 % | -0.38 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 3.39 | 3.39 | +0.00 | 3.39 | +0.00 |
| No Eligible (%) | 21.00 % | 58.43 % | +37.43 pp | 19.74 % | -1.26 pp |
| IPC Active | 3.14 inst/cycle | 1.49 inst/cycle | -1.65 | 3.12 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  7330 ns  (1.000×)
Iter 1     6530 ns  (1.123×)  — fix_tile_sizes_eliminate_partial_wave_and_thread_waste

Final best: 6530 ns  (1.1225×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 57344
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
     y0 = (yindex % 1792)
     y1 = yindex // 1792
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 1792*x2 + 16128*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 1792*x2 + 16128*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 57344
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225[grid](arg_0, arg_1, 57344, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225[grid](arg_0, arg_1, 57344, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
