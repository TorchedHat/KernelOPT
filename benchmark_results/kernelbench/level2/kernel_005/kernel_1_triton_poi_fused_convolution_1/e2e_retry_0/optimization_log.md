# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286020 ns — **Best latency**: 88740 ns (**3.2231×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_memory_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_005/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286020 ns
- **SM Throughput**: 19.72 % | **Memory**: 78.85 % | **DRAM**: 72.31 %
- **Achieved Occupancy**: 94.08 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalesced_memory_access | 3.902× | 3.223× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286020 ns | 88740 ns | -197280 ns (-69.0%) |
| SM Throughput (%) | 19.72 % | 18.95 % | -0.77 pp |
| Memory Throughput (%) | 78.85 % | 14.78 % | -64.07 pp |
| DRAM Throughput (%) | 72.31 % | 1.13 % | -71.18 pp |
| Achieved Occupancy (%) | 94.08 % | 25.00 % | -69.08 pp |
| Registers/Thread | 25.00 register/thread | 16.00 register/thread | -9 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.09 % | 73.09 % | -7.00 pp |
| IPC Active | 0.78 inst/cycle | 0.95 inst/cycle | +0.17 |

---

## Cumulative Progress

```
Baseline:  286020 ns  (1.000×)
Iter 1     88740 ns  (3.223×)  — autotune_tile_sizes_for_coalesced_memory_access

Final best: 88740 ns  (3.2231×)
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
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 2,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4),
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
