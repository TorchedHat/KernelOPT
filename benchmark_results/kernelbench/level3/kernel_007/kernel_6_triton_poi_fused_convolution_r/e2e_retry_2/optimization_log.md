# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3580 ns — **Best latency**: 3360 ns (**1.0655×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_eliminate_thread_waste`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_6_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3580 ns
- **SM Throughput**: 21.81 % | **Memory**: 20.65 % | **DRAM**: 2.57 %
- **Achieved Occupancy**: 51.06 %
- **IPC Active**: 1.92 inst/cycle | **No Eligible**: 46.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_eliminate_thread_waste | 1.071× | 1.065× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_el... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3580 ns | 3360 ns | -220 ns (-6.1%) |
| SM Throughput (%) | 21.81 % | 30.97 % | +9.16 pp |
| Memory Throughput (%) | 20.65 % | 21.06 % | +0.41 pp |
| DRAM Throughput (%) | 2.57 % | 2.75 % | +0.18 pp |
| Achieved Occupancy (%) | 51.06 % | 49.95 % | -1.11 pp |
| Registers/Thread | 28.00 register/thread | 27.00 register/thread | -1 |
| Waves/SM | 0.73 | 0.73 | +0.00 |
| No Eligible (%) | 46.69 % | 36.75 % | -9.94 pp |
| IPC Active | 1.92 inst/cycle | 2.60 inst/cycle | +0.68 |

---

## Cumulative Progress

```
Baseline:  3580 ns  (1.000×)
Iter 1     3360 ns  (1.065×)  — fix_tile_sizes_eliminate_thread_waste

Final best: 3360 ns  (1.0655×)
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
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +34,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
