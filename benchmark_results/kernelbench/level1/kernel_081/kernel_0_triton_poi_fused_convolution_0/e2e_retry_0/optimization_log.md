# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 3200 ns (**1.8188×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_block_sizes_and_warps_for_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_081/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 16.32 % | **Memory**: 42.79 % | **DRAM**: 21.10 %
- **Achieved Occupancy**: 56.08 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.397× | 1.732× | ✓ |  |
| B | autotune_block_sizes_and_warps_for_latency_hiding | 1.413× | 1.819× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 3360 ns | -2460 ns (-42.3%) | 3200 ns | -2620 ns (-45.0%) |
| SM Throughput (%) | 16.32 % | 17.27 % | +0.95 pp | 14.92 % | -1.40 pp |
| Memory Throughput (%) | 42.79 % | 20.41 % | -22.38 pp | 21.31 % | -21.48 pp |
| DRAM Throughput (%) | 21.10 % | 4.97 % | -16.13 pp | 5.18 % | -15.92 pp |
| Achieved Occupancy (%) | 56.08 % | 44.24 % | -11.84 pp | 46.96 % | -9.12 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.64 % | 54.04 % | -16.60 pp | 61.38 % | -9.26 pp |
| IPC Active | 1.11 inst/cycle | 1.71 inst/cycle | +0.60 | 1.47 inst/cycle | +0.36 |

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     3200 ns  (1.819×)  — autotune_block_sizes_and_warps_for_latency_hiding

Final best: 3200 ns  (1.8188×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 2}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=2, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=2, num_stages=1),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +41,13 @@

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
