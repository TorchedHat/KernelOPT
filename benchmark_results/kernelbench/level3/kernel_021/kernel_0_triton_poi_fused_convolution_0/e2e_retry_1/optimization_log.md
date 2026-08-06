# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6180 ns — **Best latency**: 3420 ns (**1.8070×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_tile_size_tuning`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6180 ns
- **SM Throughput**: 17.61 % | **Memory**: 39.68 % | **DRAM**: 19.93 %
- **Achieved Occupancy**: 54.74 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_tile_size_tuning | 1.343× | 1.807× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.340× | 1.507× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6180 ns | 3420 ns | -2760 ns (-44.7%) | 4100 ns | -2080 ns (-33.7%) |
| SM Throughput (%) | 17.61 % | 17.33 % | -0.28 pp | 27.33 % | +9.72 pp |
| Memory Throughput (%) | 39.68 % | 20.03 % | -19.65 pp | 21.73 % | -17.95 pp |
| DRAM Throughput (%) | 19.93 % | 4.85 % | -15.08 pp | 4.04 % | -15.89 pp |
| Achieved Occupancy (%) | 54.74 % | 45.90 % | -8.84 pp | 68.18 % | +13.44 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.29 % | 54.84 % | -15.45 pp | 41.59 % | -28.70 pp |
| IPC Active | 1.16 inst/cycle | 1.59 inst/cycle | +0.43 | 2.38 inst/cycle | +1.22 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_occupancy_and_coalescing` (1.507×)

When a Triton kernel has fewer than 1 wave/SM (grid too small to fill the GPU), adding @triton.autotune over XBLOCK/YBLOCK configurations allows the runtime to select tile sizes that increase waves/SM and achieved occupancy. In this case, doubling block size from 128 to 256 threads raised waves/SM from 0.74 to 1.48, improved achieved occupancy from 54.7% to 68.2%, and reduced warp cycles per issued instruction from 29.3 to 17.3, yielding a 1.51x speedup. The autotuner also reduces register pressure (26 → 18 registers/thread) by selecting configs with larger num_warps, further improving scheduling efficiency.

> **Applicable when**: Triton kernel has waves_per_sm < 1.0, achieved occupancy < 60%, No Eligible warp % > 60%, and CPIStall from L1TEX latency > 40% of stall cycles. Grid size is fixed (not dynamically shaped) so autotuner cost is amortized. XBLOCK and YBLOCK are tl.constexpr parameters.

> **Avoid when**: Grid is already > 2 waves/SM and occupancy > 80%. Problem sizes vary widely at runtime (autotuner overhead dominates). Kernel body has correctness constraints that restrict valid XBLOCK/YBLOCK values (e.g., shared memory sized to exact tile dimensions). Uncoalesced access pattern is inherent to the algorithm and cannot be improved by tile size alone.

---

## Cumulative Progress

```
Baseline:  6180 ns  (1.000×)
Iter 1     3420 ns  (1.807×)  — vectorized_loads_and_tile_size_tuning

Final best: 3420 ns  (1.8070×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 1024,'YBLOCK': 1}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,10 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting to 2D
+    xarange = tl.arange(0, XBLOCK)
+    xarange = tl.max_contiguous(tl.multiple_of(xarange, XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +39,13 @@

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
