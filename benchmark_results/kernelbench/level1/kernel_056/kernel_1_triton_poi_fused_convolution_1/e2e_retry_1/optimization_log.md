# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286460 ns — **Best latency**: 88740 ns (**3.2281×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`coalesce_strided_load_via_tile_swap_and_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_056/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286460 ns
- **SM Throughput**: 19.55 % | **Memory**: 78.73 % | **DRAM**: 72.04 %
- **Achieved Occupancy**: 94.03 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesce_strided_load_via_tile_swap_and_autotune | 3.162× | 3.228× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.000× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_strided_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 88740 ns | -197720 ns (-69.0%) | 284860 ns | -1600 ns (-0.6%) |
| SM Throughput (%) | 19.55 % | 18.87 % | -0.68 pp | 19.67 % | +0.12 pp |
| Memory Throughput (%) | 78.73 % | 14.82 % | -63.91 pp | 79.17 % | +0.44 pp |
| DRAM Throughput (%) | 72.04 % | 1.14 % | -70.90 pp | 72.45 % | +0.41 pp |
| Achieved Occupancy (%) | 94.03 % | 24.85 % | -69.18 pp | 94.01 % | -0.02 pp |
| Registers/Thread | 25.00 register/thread | 16.00 register/thread | -9 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.82 % | 73.16 % | -6.66 pp | 79.66 % | -0.16 pp |
| IPC Active | 0.81 inst/cycle | 0.95 inst/cycle | +0.14 | 0.81 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  286460 ns  (1.000×)
Iter 1     88740 ns  (3.228×)  — coalesce_strided_load_via_tile_swap_and_autotune

Final best: 88740 ns  (3.2281×)
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
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1,   'YBLOCK': 256}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=3),
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
@@ -19,10 +33,8 @@

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
@@ -30,7 +42,7 @@

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
