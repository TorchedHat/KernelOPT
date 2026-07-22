# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 3330 ns (**1.7778×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_084/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 15.96 % | **Memory**: 41.94 % | **DRAM**: 20.87 %
- **Achieved Occupancy**: 55.83 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 72.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.396× | 1.778× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.372× | 1.682× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3330 ns | -2590 ns (-43.8%) | 3520 ns | -2400 ns (-40.5%) |
| SM Throughput (%) | 15.96 % | 17.24 % | +1.28 pp | 25.06 % | +9.10 pp |
| Memory Throughput (%) | 41.94 % | 20.71 % | -21.23 pp | 19.44 % | -22.50 pp |
| DRAM Throughput (%) | 20.87 % | 4.99 % | -15.88 pp | 4.71 % | -16.16 pp |
| Achieved Occupancy (%) | 55.83 % | 44.17 % | -11.66 pp | 59.95 % | +4.12 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 72.07 % | 56.55 % | -15.52 pp | 40.77 % | -31.30 pp |
| IPC Active | 1.10 inst/cycle | 1.65 inst/cycle | +0.55 | 1.98 inst/cycle | +0.88 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     3330 ns  (1.778×)  — vectorized_loads_with_contiguous_hints

Final best: 3330 ns  (1.7778×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +22,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +31,13 @@

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
