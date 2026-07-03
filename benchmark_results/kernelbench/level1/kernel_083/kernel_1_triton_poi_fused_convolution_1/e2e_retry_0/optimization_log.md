# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 287070 ns — **Best latency**: 143170 ns (**2.0051×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_083/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 287070 ns
- **SM Throughput**: 19.62 % | **Memory**: 78.58 % | **DRAM**: 71.88 %
- **Achieved Occupancy**: 94.01 %
- **IPC Active**: 0.80 inst/cycle | **No Eligible**: 79.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalesced_transpose | 2.780× | 2.005× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 287070 ns | 143170 ns | -143900 ns (-50.1%) |
| SM Throughput (%) | 19.62 % | 47.82 % | +28.20 pp |
| Memory Throughput (%) | 78.58 % | 41.71 % | -36.87 pp |
| DRAM Throughput (%) | 71.88 % | 34.31 % | -37.57 pp |
| Achieved Occupancy (%) | 94.01 % | 91.98 % | -2.03 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.79 % | 52.95 % | -26.84 pp |
| IPC Active | 0.80 inst/cycle | 1.85 inst/cycle | +1.05 |

---

## Cumulative Progress

```
Baseline:  287070 ns  (1.000×)
Iter 1     143170 ns  (2.005×)  — autotune_tile_sizes_for_coalesced_transpose

Final best: 143170 ns  (2.0051×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=8),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},   num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
