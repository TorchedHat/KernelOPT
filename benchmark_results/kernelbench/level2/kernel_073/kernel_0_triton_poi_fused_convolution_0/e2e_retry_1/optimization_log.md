# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6370 ns — **Best latency**: 3360 ns (**1.8958×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_073/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6370 ns
- **SM Throughput**: 17.28 % | **Memory**: 39.86 % | **DRAM**: 19.42 %
- **Achieved Occupancy**: 55.45 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.318× | 1.431× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_occupancy_and_coalescing | 1.424× | 1.896× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.260× | 1.053× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6370 ns | 4450 ns | -1920 ns (-30.1%) | 3360 ns | -3010 ns (-47.3%) | 6050 ns | -320 ns (-5.0%) |
| SM Throughput (%) | 17.28 % | 20.56 % | +3.28 pp | 17.55 % | +0.27 pp | 16.77 % | -0.51 pp |
| Memory Throughput (%) | 39.86 % | 31.82 % | -8.04 pp | 20.40 % | -19.46 pp | 41.77 % | +1.91 pp |
| DRAM Throughput (%) | 19.42 % | 14.90 % | -4.52 pp | 4.94 % | -14.48 pp | 20.30 % | +0.88 pp |
| Achieved Occupancy (%) | 55.45 % | 46.69 % | -8.76 pp | 43.86 % | -11.59 pp | 55.38 % | -0.07 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.13 % | 55.85 % | -15.28 pp | 55.66 % | -15.47 pp | 70.04 % | -1.09 pp |
| IPC Active | 1.15 inst/cycle | 1.68 inst/cycle | +0.53 | 1.68 inst/cycle | +0.53 | 1.14 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguity_hints` (1.431×)

When a Triton kernel loads from a globally contiguous address range (stride-1 in the innermost dimension) but lacks contiguity hints, the compiler emits scalar 32-bit loads, causing high L1TEX stall cycles (CPIStall). Wrapping the innermost arange with tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK) proves to the compiler that the offset is XBLOCK-aligned and the range is contiguous, enabling LDG.128 (128-bit vectorized) loads. This reduced warp cycles per issued instruction from 30.15 to 16.97 and raised issue slot utilization from 29.3% to 43.2%, yielding a 1.43x speedup. Pairing with @triton.autotune over XBLOCK/YBLOCK/num_warps allows the runtime to select the best tile configuration.

> **Applicable when**: Triton kernel loads from a stride-1 (contiguous) global memory region in the innermost dimension; CPIStall rule is triggered with L1TEX stall cycles > 10 cycles/instruction; UncoalescedGlobalAccess rule fires with significant excessive sectors; Issue Slots Busy < 35%; Eligible Warps Per Scheduler < 1.5.

> **Avoid when**: The load access pattern is non-contiguous or strided in the innermost dimension (e.g., transposed access, gather patterns); the kernel is already compute-bound rather than memory-latency-bound; XBLOCK is not a compile-time constant aligned to the program_id stride.

---

## Cumulative Progress

```
Baseline:  6370 ns  (1.000×)
Iter 1     3360 ns  (1.896×)  — autotune_tile_sizes_and_warps_for_occupancy_and_coalescing

Final best: 3360 ns  (1.8958×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
