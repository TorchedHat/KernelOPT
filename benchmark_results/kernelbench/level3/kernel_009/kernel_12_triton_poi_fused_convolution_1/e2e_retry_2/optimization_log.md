# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 2880 ns (**1.2222×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_xnumel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3520 ns
- **SM Throughput**: 22.77 % | **Memory**: 21.04 % | **DRAM**: 2.60 %
- **Achieved Occupancy**: 50.96 %
- **IPC Active**: 1.99 inst/cycle | **No Eligible**: 45.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_small_xnumel | 1.100× | 1.222× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3520 ns | 2880 ns | -640 ns (-18.2%) |
| SM Throughput (%) | 22.77 % | 8.01 % | -14.76 pp |
| Memory Throughput (%) | 21.04 % | 3.19 % | -17.85 pp |
| DRAM Throughput (%) | 2.60 % | 3.19 % | +0.59 pp |
| Achieved Occupancy (%) | 50.96 % | 37.60 % | -13.36 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.73 | +0.00 |
| No Eligible (%) | 45.94 % | 66.48 % | +20.54 pp |
| IPC Active | 1.99 inst/cycle | 1.10 inst/cycle | -0.89 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_small_xnumel` (1.222×)

When a Triton kernel's XBLOCK far exceeds xnumel (e.g., XBLOCK=128 for xnumel=9), the vast majority of threads in the X-dimension are masked out, wasting register and shared-memory resources and inflating instruction count. Replacing the hardcoded oversized XBLOCK with an @triton.autotune decorator that explores small XBLOCK values (1–16) matched to the actual xnumel reduces executed instructions by ~73% (638K→173K) and cuts register usage from 28 to 16 per thread, eliminating unnecessary dynamic shared memory allocation and yielding a 1.22x speedup. The grid shape and waves-per-SM may not change dramatically if ynumel already drives the grid, but the reduction in wasted work per CTA directly shortens kernel duration.

> **Applicable when**: xnumel is much smaller than the hardcoded XBLOCK (e.g., xnumel <= 16 but XBLOCK=128), causing most threads in the X-dimension to be masked out; MemoryCacheAccessPattern rule fires with low bytes-utilized-per-sector; IssueSlotUtilization shows high No-Eligible percentage; executed instruction count is inflated relative to actual work

> **Avoid when**: xnumel is close to or larger than XBLOCK (waste is minimal); the kernel is already autotuned; the grid is already large enough to saturate the GPU (waves_per_sm >> 1); changing tile sizes would break index arithmetic correctness

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 1     2880 ns  (1.222×)  — autotune_tile_sizes_for_small_xnumel

Final best: 2880 ns  (1.2222×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +40,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
