# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 4380 ns (**1.3881×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_contiguity_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_073/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 16.08 % | **Memory**: 41.80 % | **DRAM**: 20.37 %
- **Achieved Occupancy**: 56.74 %
- **IPC Active**: 1.04 inst/cycle | **No Eligible**: 70.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.294× | 1.388× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6080 ns | 4380 ns | -1700 ns (-28.0%) |
| SM Throughput (%) | 16.08 % | 22.01 % | +5.93 pp |
| Memory Throughput (%) | 41.80 % | 32.05 % | -9.75 pp |
| DRAM Throughput (%) | 20.37 % | 14.98 % | -5.39 pp |
| Achieved Occupancy (%) | 56.74 % | 47.56 % | -9.18 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.72 % | 56.99 % | -13.73 pp |
| IPC Active | 1.04 inst/cycle | 1.67 inst/cycle | +0.63 |

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     4380 ns  (1.388×)  — vectorized_loads_with_contiguity_hints

Final best: 4380 ns  (1.3881×)
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
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +25,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting to 2D
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), 16), 16)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +36,13 @@

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
