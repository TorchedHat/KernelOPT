# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 5250 ns (**1.1524×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_improve_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_078/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 16.37 % | **Memory**: 41.23 % | **DRAM**: 20.36 %
- **Achieved Occupancy**: 55.08 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_improve_coalescing_and_occupancy | 1.260× | 1.152× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6050 ns | 5250 ns | -800 ns (-13.2%) |
| SM Throughput (%) | 16.37 % | 27.17 % | +10.80 pp |
| Memory Throughput (%) | 41.23 % | 29.78 % | -11.45 pp |
| DRAM Throughput (%) | 20.36 % | 3.15 % | -17.21 pp |
| Achieved Occupancy (%) | 55.08 % | 73.51 % | +18.43 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 1.48 | +0.74 |
| No Eligible (%) | 71.28 % | 37.67 % | -33.61 pp |
| IPC Active | 1.14 inst/cycle | 2.44 inst/cycle | +1.30 |

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 1     5250 ns  (1.152×)  — autotune_tile_sizes_to_improve_coalescing_and_occupancy

Final best: 5250 ns  (1.1524×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +33,13 @@

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
