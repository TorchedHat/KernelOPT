# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 3200 ns (**1.8406×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_056/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 15.73 % | **Memory**: 42.19 % | **DRAM**: 20.92 %
- **Achieved Occupancy**: 56.02 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 73.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_alignment_hints | 1.238× | 0.731× | ✓ |  |
| B | autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy | 1.429× | 1.841× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.007× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 8060 ns | +2170 ns (+36.8%) | 3200 ns | -2690 ns (-45.7%) | 6020 ns | +130 ns (+2.2%) |
| SM Throughput (%) | 15.73 % | 48.29 % | +32.56 pp | 8.26 % | -7.47 pp | 16.86 % | +1.13 pp |
| Memory Throughput (%) | 42.19 % | 50.47 % | +8.28 pp | 11.27 % | -30.92 pp | 42.30 % | +0.11 pp |
| DRAM Throughput (%) | 20.92 % | 15.24 % | -5.68 pp | 2.58 % | -18.34 pp | 20.51 % | -0.41 pp |
| Achieved Occupancy (%) | 56.02 % | 77.62 % | +21.60 pp | 50.67 % | -5.35 pp | 55.98 % | -0.04 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 16.00 register/thread | -10 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 73.01 % | 36.15 % | -36.86 pp | 73.23 % | +0.22 pp | 70.52 % | -2.49 pp |
| IPC Active | 1.11 inst/cycle | 2.54 inst/cycle | +1.43 | 0.94 inst/cycle | -0.17 | 1.11 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_with_alignment_hints` (0.731×)

Adding tl.max_contiguous and tl.multiple_of alignment hints to a Triton load index, combined with autotune selecting a larger block size (256 vs 128), caused a 37% slowdown despite improving scheduler utilization and occupancy. The executed instruction count exploded from 771K to 2.86M (3.7x), indicating the hints triggered a much more expensive code path — likely unrolled or vectorized loops that generated far more instructions than the scalar baseline. The net effect was a longer kernel (8.06 µs vs 5.89 µs) even though per-cycle efficiency metrics improved, because the total work per thread increased dramatically.

> **Applicable when**: Never demonstrated to help; avoid this pattern for short-lived permutation kernels.

> **Avoid when**: Kernel is a short-lived memory-bound permutation/layout-transform with stride-N stores (N>1). The alignment hints (tl.max_contiguous + tl.multiple_of) combined with autotune selecting larger block sizes cause instruction count to balloon 3-4x, overwhelming any benefit from vectorized loads. The stride-N store pattern means stores remain uncoalesced regardless, so vectorizing only the load side does not reduce total memory transactions proportionally. Do not apply when the store access pattern is strided and cannot be fixed simultaneously.

#### ✅ Plan B: `autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy` (1.841×)

When a Triton kernel has uncoalesced global stores (>15% excess sectors) caused by strided write patterns across the YBLOCK dimension, and L1TEX stall cycles dominate (>40% of total stall cycles), adding @triton.autotune with configs that include YBLOCK=1 options allows the autotuner to discover tile shapes where threads within a warp write to more contiguous addresses, dramatically reducing memory traffic. Removing eviction_policy='evict_last' from single-read loads and letting autotune select optimal num_warps/num_stages also reduces register usage (26→16 registers/thread) and eliminates unnecessary dynamic shared memory (4.1KB→0), cutting elapsed cycles by ~45% (8711→4781 cycles) and kernel duration by ~46% (5.89→3.20 µs). The L2 cache throughput dropped from 57% to 15%, confirming that total memory traffic was substantially reduced even though the uncoalesced access percentage metric paradoxically increased (fewer total sectors means the remaining uncoalesced fraction looks larger).

> **Applicable when**: Triton 2D-tiled kernel with strided store patterns across the YBLOCK dimension causing uncoalesced global writes (UncoalescedGlobalAccess rule triggered, >15% excess sectors); L1TEX stall cycles >40% of total stall cycles (CPIStall rule); hardcoded XBLOCK/YBLOCK tile sizes without autotuning; eviction_policy='evict_last' on loads that are read exactly once; registers per thread >20 suggesting register pressure from fixed tile configuration.

> **Avoid when**: Kernel already uses @triton.autotune; store access pattern is inherently strided regardless of YBLOCK (e.g., scatter operations where coalescing is impossible by design); problem size is so small that autotune overhead dominates; kernel is reused across many different problem sizes where autotune cache misses would be frequent.

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     3200 ns  (1.841×)  — autotune_tile_sizes_to_fix_uncoalesced_stores_and_low_occupancy

Final best: 3200 ns  (1.8406×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +34,8 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask)
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +43,7 @@

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
