# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 3360 ns (**1.7321×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 15.97 % | **Memory**: 42.79 % | **DRAM**: 21.21 %
- **Achieved Occupancy**: 55.76 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.398× | 1.732× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_for_coalescing_and_occupancy | 1.279× | 1.357× | ✓ |  |
| C | add_triton_autotune | 1.024× | 0.957× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 3360 ns | -2460 ns (-42.3%) | 4290 ns | -1530 ns (-26.3%) | 6080 ns | +260 ns (+4.5%) |
| SM Throughput (%) | 15.97 % | 17.26 % | +1.29 pp | 26.11 % | +10.14 pp | 16.80 % | +0.83 pp |
| Memory Throughput (%) | 42.79 % | 20.49 % | -22.30 pp | 20.74 % | -22.05 pp | 42.07 % | -0.72 pp |
| DRAM Throughput (%) | 21.21 % | 4.94 % | -16.27 pp | 3.86 % | -17.35 pp | 20.34 % | -0.87 pp |
| Achieved Occupancy (%) | 55.76 % | 43.66 % | -12.10 pp | 68.57 % | +12.81 pp | 56.02 % | +0.26 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.58 % | 55.01 % | -15.57 pp | 38.17 % | -32.41 pp | 72.90 % | +2.32 pp |
| IPC Active | 1.12 inst/cycle | 1.67 inst/cycle | +0.55 | 2.41 inst/cycle | +1.29 | 1.12 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     3360 ns  (1.732×)  — vectorized_loads_with_autotune

Final best: 3360 ns  (1.7321×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +25,8 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex_1d = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xindex_1d[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +35,13 @@

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
