# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 6020 ns (**0.9784×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 16.41 % | **Memory**: 41.53 % | **DRAM**: 20.93 %
- **Achieved Occupancy**: 54.67 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.250× | 0.978× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.084× | 0.588× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 6020 ns | +130 ns (+2.2%) | 10020 ns | +4130 ns (+70.1%) |
| SM Throughput (%) | 16.41 % | 16.63 % | +0.22 pp | 14.77 % | -1.64 pp |
| Memory Throughput (%) | 41.53 % | 40.45 % | -1.08 pp | 34.17 % | -7.36 pp |
| DRAM Throughput (%) | 20.93 % | 20.47 % | -0.46 pp | 6.14 % | -14.79 pp |
| Achieved Occupancy (%) | 54.67 % | 54.64 % | -0.03 pp | 52.64 % | -2.03 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.96 % | 71.86 % | +0.90 pp | 74.02 % | +3.06 pp |
| IPC Active | 1.14 inst/cycle | 1.14 inst/cycle | +0.00 | 0.98 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)

Final best: 6020 ns  (0.9784×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +25,10 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(
+        tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK),
+        XBLOCK
+    )[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
