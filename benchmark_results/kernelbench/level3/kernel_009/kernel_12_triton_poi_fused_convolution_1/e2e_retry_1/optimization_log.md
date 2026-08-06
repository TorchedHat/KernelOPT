# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 2880 ns (**1.2222×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_fix_xblock_waste`** beats baseline latency.

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
- **SM Throughput**: 22.73 % | **Memory**: 21.01 % | **DRAM**: 2.62 %
- **Achieved Occupancy**: 53.66 %
- **IPC Active**: 2.09 inst/cycle | **No Eligible**: 51.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_fix_xblock_waste | 1.062× | 1.222× | ✓ | **✓ best** |
| C | add_triton_autotune | 0.677× | 0.534× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2880 ns | -640 ns (-18.2%) | 6590 ns | +3070 ns (+87.2%) |
| SM Throughput (%) | 22.73 % | 8.15 % | -14.58 pp | 42.42 % | +19.69 pp |
| Memory Throughput (%) | 21.01 % | 3.18 % | -17.83 pp | 50.45 % | +29.44 pp |
| DRAM Throughput (%) | 2.62 % | 3.18 % | +0.56 pp | 1.40 % | -1.22 pp |
| Achieved Occupancy (%) | 53.66 % | 41.69 % | -11.97 pp | 45.83 % | -7.83 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 48.00 register/thread | +20 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 1.16 | +0.43 |
| No Eligible (%) | 51.92 % | 65.79 % | +13.87 pp | 35.89 % | -16.03 pp |
| IPC Active | 2.09 inst/cycle | 0.94 inst/cycle | -1.15 | 2.54 inst/cycle | +0.45 |


### Key Learnings

#### ✅ Plan A: `autotune_tile_sizes_fix_xblock_waste` (1.222×)

When a Triton kernel's XBLOCK is much larger than xnumel (e.g., XBLOCK=128 for xnumel=9), the vast majority of threads in the X-dimension are masked out and do no useful work, wasting register and shared-memory resources. Replacing the fixed oversized XBLOCK with @triton.autotune over small XBLOCK values (1, 4, 8, 16) matching the actual xnumel size eliminates this idle-thread waste: registers per thread dropped from 28 to 16, dynamic shared memory dropped from 4 KB to 0, and executed instructions dropped from 638K to 173K, yielding a 1.22x speedup. The grid size and waves/SM remained the same (0.73), indicating the speedup came from reduced per-thread overhead rather than improved SM saturation.

> **Applicable when**: Triton kernel has a small xnumel (e.g., <=16) but XBLOCK is set to a large default (e.g., 128); registers_per_thread is high (>20); dynamic shared memory is allocated unnecessarily; executed instruction count is much higher than expected for the workload size; IssueSlotUtilization and AchievedOccupancy rules are triggered with large gaps between theoretical and achieved occupancy.

> **Avoid when**: xnumel is large (e.g., >= XBLOCK/2) so thread waste is minimal; the kernel is already autotuned; adding autotune overhead (compilation of multiple configs) is unacceptable in latency-sensitive single-use contexts; the grid is already saturating SMs (waves_per_sm >> 1).

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 1     2880 ns  (1.222×)  — autotune_tile_sizes_fix_xblock_waste

Final best: 2880 ns  (1.2222×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +36,13 @@

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
