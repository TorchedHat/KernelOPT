# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_47` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2530 ns — **Best latency**: 2750 ns (**0.9200×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_47`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_47_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (192,1,1), Waves/SM: 0.09
- **Registers/thread**: 28
- **Duration**: 2530 ns
- **SM Throughput**: 3.46 % | **Memory**: 3.19 % | **DRAM**: 1.28 %
- **Achieved Occupancy**: 8.72 %
- **IPC Active**: 0.47 inst/cycle | **No Eligible**: 88.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.988× | 0.542× | ✓ |  |
| B | autotune_tile_sizes_to_fix_grid_underutilization | 1.054× | 0.920× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 4670 ns | +2140 ns (+84.6%) | 2750 ns | +220 ns (+8.7%) |
| SM Throughput (%) | 3.46 % | 1.05 % | -2.41 pp | 4.13 % | +0.67 pp |
| Memory Throughput (%) | 3.19 % | 1.38 % | -1.81 pp | 3.92 % | +0.73 pp |
| DRAM Throughput (%) | 1.28 % | 0.69 % | -0.59 pp | 1.18 % | -0.10 pp |
| Achieved Occupancy (%) | 8.72 % | 7.14 % | -1.58 pp | 8.63 % | -0.09 pp |
| Registers/Thread | 28.00 register/thread | 25.00 register/thread | -3 | 28.00 register/thread | +0 |
| Waves/SM | 0.09 | 0.09 | +0.00 | 0.09 | +0.00 |
| No Eligible (%) | 88.74 % | 93.40 % | +4.66 pp | 82.24 % | -6.50 pp |
| IPC Active | 0.47 inst/cycle | 0.25 inst/cycle | -0.22 | 0.53 inst/cycle | +0.06 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune` (0.542×)

Adding a persistent while-loop with autotune to a small-grid kernel (192 CTAs, 0.09 waves/SM) caused a 1.85x slowdown despite the intent to improve occupancy. The after kernel shows elapsed cycles nearly doubling (3768→6966), achieved occupancy dropping further (8.72%→7.14%), and warp cycles per issued instruction worsening (12.7→17.82). The persistent loop introduced additional branch overhead and L1TEX stalls (CPIStall shifted from barrier-wait to L1TEX dependency, 31.6%→31.1%), while the autotune grid-capping logic left the launcher still using the old fixed grid (cdiv(25,128), cdiv(1536,8)) = (1,192), negating any persistent-loop benefit. For kernels this small (total work ~38K elements), the overhead of the while-loop control flow and autotune compilation cost exceeds any scheduling gain.

> **Applicable when**: Never: this specific combination of persistent loop + autotune on a tiny-grid kernel caused regression. The pattern may work for larger grids where the loop overhead is amortized.

> **Avoid when**: Kernel has very small total element count (<100K elements), grid is already near-minimal (waves_per_sm < 0.1), and the launcher grid is not updated to match the persistent-loop intent. The while-loop control flow adds branch overhead that dominates for short-running kernels. Also avoid when the autotune key does not change between calls (constant ynumel/xnumel), as autotune overhead is paid every cold start.

---

## Cumulative Progress

```
Baseline:  2530 ns  (1.000×)

Final best: 2750 ns  (0.9200×)
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
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1536
@@ -22,15 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 24*x2 + 600*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 1536
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_47[grid](arg_0, arg_1, 1536, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_47[grid](arg_0, arg_1, 1536, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
