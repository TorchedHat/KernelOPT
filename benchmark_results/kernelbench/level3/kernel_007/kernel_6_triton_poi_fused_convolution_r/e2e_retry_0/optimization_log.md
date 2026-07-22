# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3620 ns — **Best latency**: 3140 ns (**1.1529×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_xnumel`** beats baseline latency.

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
- **Duration**: 3620 ns
- **SM Throughput**: 22.22 % | **Memory**: 21.07 % | **DRAM**: 2.54 %
- **Achieved Occupancy**: 51.65 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 46.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_small_xnumel | 1.117× | 1.153× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.004× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 3140 ns | -480 ns (-13.3%) | 3550 ns | -70 ns (-1.9%) |
| SM Throughput (%) | 22.22 % | 8.19 % | -14.03 pp | 21.49 % | -0.73 pp |
| Memory Throughput (%) | 21.07 % | 2.92 % | -18.15 pp | 20.32 % | -0.75 pp |
| DRAM Throughput (%) | 2.54 % | 2.92 % | +0.38 pp | 2.58 % | +0.04 pp |
| Achieved Occupancy (%) | 51.65 % | 36.00 % | -15.65 pp | 51.59 % | -0.06 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 46.72 % | 65.14 % | +18.42 pp | 48.59 % | +1.87 pp |
| IPC Active | 1.93 inst/cycle | 1.11 inst/cycle | -0.82 | 1.96 inst/cycle | +0.03 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_small_xnumel` (1.153×)

When a 2D Triton kernel has a very small xnumel (e.g., 9) but a large hardcoded XBLOCK (e.g., 128), the vast majority of threads in the X dimension are masked out and do no useful work, wasting thread slots and inflating register/shared-memory usage. Adding @triton.autotune with configs that include small XBLOCK values (1, 4, 8) and larger YBLOCK values allows the runtime to select tile shapes that eliminate this waste, reducing executed instructions (620K→173K) and register pressure (28→16 registers/thread). The speedup here was modest (1.15x) because waves_per_sm remained at 0.73 — the grid lambda was not updated to use meta['XBLOCK']/meta['YBLOCK'], so the grid shape did not change with the autotuned config, limiting the benefit to register/smem savings only.

> **Applicable when**: Triton 2D pointwise kernel where xnumel << XBLOCK (e.g., xnumel=9, XBLOCK=128), causing >50% of X-dimension threads to be masked; waves_per_sm < 1.0; achieved occupancy significantly below theoretical; IssueSlotUtilization rule triggered with no_eligible_warp_pct > 40%.

> **Avoid when**: xnumel is close to or larger than XBLOCK (waste is minimal); kernel is already autotuned; grid is not updated to use meta['XBLOCK']/meta['YBLOCK'] — without the lambda grid, the tile shape change does not affect grid dimensions and the occupancy/wave benefit is lost. Do not apply when the kernel has no Y dimension or when YBLOCK changes would break shared memory tiling logic.

---

## Cumulative Progress

```
Baseline:  3620 ns  (1.000×)
Iter 1     3140 ns  (1.153×)  — autotune_tile_sizes_for_small_xnumel

Final best: 3140 ns  (1.1529×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=4),
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

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
