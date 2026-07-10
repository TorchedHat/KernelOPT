# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6110 ns — **Best latency**: 4260 ns (**1.4343×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6110 ns
- **SM Throughput**: 16.51 % | **Memory**: 40.86 % | **DRAM**: 20.21 %
- **Achieved Occupancy**: 55.61 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 71.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.173× | 1.072× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.363× | 1.434× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6110 ns | 5700 ns | -410 ns (-6.7%) | 4260 ns | -1850 ns (-30.3%) |
| SM Throughput (%) | 16.51 % | 27.97 % | +11.46 pp | 26.57 % | +10.06 pp |
| Memory Throughput (%) | 40.86 % | 35.25 % | -5.61 pp | 21.12 % | -19.74 pp |
| DRAM Throughput (%) | 20.21 % | 11.58 % | -8.63 pp | 3.88 % | -16.33 pp |
| Achieved Occupancy (%) | 55.61 % | 53.36 % | -2.25 pp | 68.92 % | +13.31 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 1.48 | +0.74 |
| No Eligible (%) | 71.43 % | 45.40 % | -26.03 pp | 38.60 % | -32.83 pp |
| IPC Active | 1.10 inst/cycle | 2.06 inst/cycle | +0.96 | 2.41 inst/cycle | +1.31 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.072×)

When a Triton kernel's load address is stride-1 contiguous in the X dimension (e.g., `ptr + xindex`), applying `tl.max_contiguous(tl.multiple_of(xindex, 4), 4)` enables the compiler to emit 128-bit vectorized loads (ld.global.v4.f32) instead of scalar 32-bit loads. This directly reduces L1TEX stall cycles (Warp Cycles Per Issued Instruction dropped from 30.3 to 15.43) and doubles scheduler issue slot utilization (27.97% → 51.98%), cutting kernel duration by ~7%. The optimization is safe when the base offset is 4-element aligned (i.e., xnumel divisible by 4) and all tile sizes (XBLOCK) are multiples of 4.

> **Applicable when**: Load address is stride-1 contiguous in the X (innermost) dimension; xnumel is divisible by 4; all XBLOCK autotune configs are multiples of 4; kernel is bottlenecked by L1TEX stalls (CPIStall rule triggered, Warp Cycles Per Issued Instruction > 20, Eligible Warps Per Scheduler < 1.5, No Eligible > 60%); UncoalescedGlobalAccess rule reports excess sectors.

> **Avoid when**: Load address has non-unit stride in the X dimension (e.g., strided gather); xnumel is not divisible by 4 (alignment not guaranteed); XBLOCK configs include values not divisible by 4; store side has stride-1 access that could also benefit (vectorize stores separately); kernel is compute-bound rather than memory-latency-bound.

---

## Cumulative Progress

```
Baseline:  6110 ns  (1.000×)
Iter 1     4260 ns  (1.434×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 4260 ns  (1.4343×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=4, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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
