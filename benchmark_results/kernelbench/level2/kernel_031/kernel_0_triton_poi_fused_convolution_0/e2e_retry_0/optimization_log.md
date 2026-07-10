# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 4130 ns (**1.4092×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_031/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 16.69 % | **Memory**: 42.40 % | **DRAM**: 21.16 %
- **Achieved Occupancy**: 55.92 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 71.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.373× | 1.409× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 1.366× | 1.608× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 4130 ns | -1690 ns (-29.0%) | 3620 ns | -2200 ns (-37.8%) |
| SM Throughput (%) | 16.69 % | 26.73 % | +10.04 pp | 10.18 % | -6.51 pp |
| Memory Throughput (%) | 42.40 % | 21.21 % | -21.19 pp | 19.85 % | -22.55 pp |
| DRAM Throughput (%) | 21.16 % | 3.98 % | -17.18 pp | 9.16 % | -12.00 pp |
| Achieved Occupancy (%) | 55.92 % | 67.94 % | +12.02 pp | 30.27 % | -25.65 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 71.93 % | 38.45 % | -33.48 pp | 74.72 % | +2.79 pp |
| IPC Active | 1.10 inst/cycle | 2.40 inst/cycle | +1.30 | 0.94 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     4130 ns  (1.409×)  — vectorized_loads_with_autotune

Final best: 4130 ns  (1.4092×)
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
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=4),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 8}, num_warps=4, num_stages=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +24,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]  # tells compiler indices are contiguous multiples of XBLOCK -> enables float4 vectorized loads
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
