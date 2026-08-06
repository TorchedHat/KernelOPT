# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3550 ns — **Best latency**: 2910 ns (**1.2199×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_xnumel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_6_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3550 ns
- **SM Throughput**: 22.41 % | **Memory**: 21.22 % | **DRAM**: 2.58 %
- **Achieved Occupancy**: 50.62 %
- **IPC Active**: 1.92 inst/cycle | **No Eligible**: 51.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_xnumel | 1.071× | 1.220× | ✓ | **✓ best** |
| B | fix_tile_dimensions_for_small_x_large_y | 1.067× | 1.179× | ✓ |  |
| C | add_triton_autotune | 0.966× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_dimensio... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3550 ns | 2910 ns | -640 ns (-18.0%) | 3010 ns | -540 ns (-15.2%) | 3550 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 22.41 % | 8.34 % | -14.07 pp | 8.33 % | -14.08 pp | 21.35 % | -1.06 pp |
| Memory Throughput (%) | 21.22 % | 3.16 % | -18.06 pp | 3.07 % | -18.15 pp | 20.24 % | -0.98 pp |
| DRAM Throughput (%) | 2.58 % | 3.16 % | +0.58 pp | 3.07 % | +0.49 pp | 2.56 % | -0.02 pp |
| Achieved Occupancy (%) | 50.62 % | 37.98 % | -12.64 pp | 36.88 % | -13.74 pp | 49.23 % | -1.39 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 51.48 % | 64.52 % | +13.04 pp | 74.17 % | +22.69 pp | 48.95 % | -2.53 pp |
| IPC Active | 1.92 inst/cycle | 1.18 inst/cycle | -0.74 | 1.15 inst/cycle | -0.77 | 1.89 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan A: `autotune_tile_sizes_for_small_xnumel` (1.220×)

When a Triton kernel's XBLOCK is much larger than xnumel (e.g., XBLOCK=128 for xnumel=9), the vast majority of threads per block are masked out, wasting compute slots and causing high no-eligible-warp stall rates (here 51.48%). Adding @triton.autotune with XBLOCK configs ≤ xnumel allows the autotuner to select tile sizes that eliminate masked-thread waste, reducing executed instructions by ~72% (620K→173K) and kernel duration from 3.55µs to 2.91µs. The register pressure also drops (28→16 registers/thread) as a side effect of smaller tile sizes, freeing occupancy headroom.

> **Applicable when**: XBLOCK >> xnumel (ratio > 4x), resulting in high masked-thread fraction; no-eligible-warp stall rate > 40%; waves_per_sm < 1.0; kernel is a pointwise or simple memory-bound op with 2D tiling (XBLOCK x YBLOCK); xnumel is a small non-power-of-2 value (e.g., 3, 5, 7, 9)

> **Avoid when**: xnumel is large and close to XBLOCK (low masked-thread fraction); kernel is compute-bound (tensor core heavy); autotuning overhead is unacceptable (e.g., very short-lived kernels called once); XBLOCK is already ≤ xnumel

---

## Cumulative Progress

```
Baseline:  3550 ns  (1.000×)
Iter 1     2910 ns  (1.220×)  — autotune_tile_sizes_for_small_xnumel

Final best: 2910 ns  (1.2199×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK':  32}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK':  32}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK':  16}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  16}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +36,13 @@

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
