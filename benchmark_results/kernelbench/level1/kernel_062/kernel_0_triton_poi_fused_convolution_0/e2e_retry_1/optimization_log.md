# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 4190 ns (**1.4057×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_062/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 16.64 % | **Memory**: 42.47 % | **DRAM**: 21.04 %
- **Achieved Occupancy**: 54.95 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.308× | 1.406× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.026× | 1.028× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4190 ns | -1700 ns (-28.9%) | 5730 ns | -160 ns (-2.7%) |
| SM Throughput (%) | 16.64 % | 19.26 % | +2.62 pp | 16.32 % | -0.32 pp |
| Memory Throughput (%) | 42.47 % | 22.27 % | -20.20 pp | 43.31 % | +0.84 pp |
| DRAM Throughput (%) | 21.04 % | 3.95 % | -17.09 pp | 21.53 % | +0.49 pp |
| Achieved Occupancy (%) | 54.95 % | 48.85 % | -6.10 pp | 55.79 % | +0.84 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.74 % | 51.71 % | -19.03 pp | 72.77 % | +2.03 pp |
| IPC Active | 1.14 inst/cycle | 1.84 inst/cycle | +0.70 | 1.12 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     4190 ns  (1.406×)  — vectorized_loads_and_autotune_tile_sizes

Final best: 4190 ns  (1.4057×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting to 2D
+    x1d = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = x1d[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +38,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
