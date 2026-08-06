# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_59` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2560 ns — **Best latency**: 3040 ns (**0.8421×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_59`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_59_triton_poi_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.12
- **Registers/thread**: 28
- **Duration**: 2560 ns
- **SM Throughput**: 4.69 % | **Memory**: 5.09 % | **DRAM**: 1.67 %
- **Achieved Occupancy**: 11.16 %
- **IPC Active**: 0.59 inst/cycle | **No Eligible**: 83.73 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotuned_block_sizes | 0.135× | 0.032× | ✓ |  |
| B | autotune_tile_sizes_for_small_grid_occupancy | 1.043× | 0.842× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 79900 ns | +77340 ns (+3021.1%) | 3040 ns | +480 ns (+18.8%) |
| SM Throughput (%) | 4.69 % | 9.42 % | +4.73 pp | 2.14 % | -2.55 pp |
| Memory Throughput (%) | 5.09 % | 16.13 % | +11.04 pp | 8.47 % | +3.38 pp |
| DRAM Throughput (%) | 1.67 % | 0.06 % | -1.61 pp | 8.47 % | +6.80 pp |
| Achieved Occupancy (%) | 11.16 % | 6.06 % | -5.10 pp | 5.70 % | -5.46 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 21.00 register/thread | -7 |
| Waves/SM | 0.12 | 0.06 | -0.06 | 0.06 | -0.06 |
| No Eligible (%) | 83.73 % | 89.64 % | +5.91 pp | 92.97 % | +9.24 pp |
| IPC Active | 0.59 inst/cycle | 0.40 inst/cycle | -0.19 | 0.24 inst/cycle | -0.35 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotuned_block_sizes` (0.032×)

Adding a persistent while-loop inside a Triton kernel to handle multiple tiles per CTA caused a 58x increase in executed instructions (102K → 5.98M) and a 31x increase in kernel duration (2.56 µs → 79.9 µs). The grid was not updated to use the persistent SM-count-based launch (it remained at the original 256 CTAs), so each CTA iterated through all tiles sequentially rather than in parallel, serializing work that was previously parallel. This anti-pattern occurs when the persistent loop is added but the grid size is not simultaneously reduced to the SM count — the original grid already covered all tiles, so the loop caused each CTA to redundantly process all tiles.

> **Applicable when**: Never apply this pattern as implemented — it is always wrong when the grid is not changed to SM_COUNT.

> **Avoid when**: Do NOT add a persistent while-loop to a Triton kernel without simultaneously changing the grid to (SM_COUNT,) or (SM_COUNT * multiplier,). If the grid remains the same as the original tile-covering grid, each CTA will iterate over all tiles, multiplying total work by num_tiles and causing catastrophic slowdown. This is especially harmful for tiny kernels (< 1000 µs) where launch overhead is already the bottleneck.

#### ❌ Plan B: `autotune_tile_sizes_for_small_grid_occupancy` (0.842×)

When a kernel has very low achieved occupancy (e.g., ~11%) due to a tile dimension (XBLOCK) far exceeding the actual problem size in that dimension, reducing XBLOCK via autotune was expected to expand the grid and improve SM utilization. However, in this case the autotune decorator was added but the grid lambda was not updated — the kernel_function still used the hardcoded old grid formula (cdiv(xnumel, 128), cdiv(ynumel, 8)) instead of a dynamic lambda reading meta['XBLOCK'] and meta['YBLOCK']. As a result, the grid remained fixed at 256 blocks regardless of the autotuned tile sizes, waves/SM dropped from 0.12 to 0.06 (halved), achieved occupancy fell from 11.2% to 5.7%, L2 hit rate dropped from 60% to 35.5% (fewer reuses due to smaller blocks), and duration increased from 2.56 µs to 3.04 µs.

> **Applicable when**: Kernel has achieved occupancy far below theoretical (e.g., <20%), waves_per_sm < 0.2, and a tile dimension (XBLOCK/YBLOCK) is much larger than the actual problem size in that dimension, causing most threads in a block to be idle.

> **Avoid when**: The grid lambda is not updated to use meta['XBLOCK'] and meta['YBLOCK'] — this implementation flaw causes the grid to remain fixed while block size shrinks, halving waves_per_sm and worsening occupancy. Also avoid when the kernel is already memory-bandwidth bound and L2 hit rate is high (smaller tiles reduce reuse).

---

## Cumulative Progress

```
Baseline:  2560 ns  (1.000×)

Final best: 3040 ns  (0.8421×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 2048
     xnumel = 25
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = tl.full([YBLOCK, 1], True, tl.int1)
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 2048
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
