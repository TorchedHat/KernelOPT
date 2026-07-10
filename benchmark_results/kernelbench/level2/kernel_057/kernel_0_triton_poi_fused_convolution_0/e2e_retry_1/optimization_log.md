# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 6210 ns (**0.9533×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.21 % | **Memory**: 43.04 % | **DRAM**: 20.90 %
- **Achieved Occupancy**: 56.02 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 71.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_pipeline_stages | 1.242× | 0.953× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5920 ns | 6210 ns | +290 ns (+4.9%) |
| SM Throughput (%) | 16.21 % | 15.65 % | -0.56 pp |
| Memory Throughput (%) | 43.04 % | 40.47 % | -2.57 pp |
| DRAM Throughput (%) | 20.90 % | 19.89 % | -1.01 pp |
| Achieved Occupancy (%) | 56.02 % | 54.49 % | -1.53 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.55 % | 71.45 % | -0.10 pp |
| IPC Active | 1.11 inst/cycle | 1.15 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)

Final best: 6210 ns  (0.9533×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
