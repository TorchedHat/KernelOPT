# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6180 ns — **Best latency**: 3360 ns (**1.8393×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_011/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6180 ns
- **SM Throughput**: 16.35 % | **Memory**: 39.99 % | **DRAM**: 20.03 %
- **Achieved Occupancy**: 54.73 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 74.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of | 1.098× | 1.177× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.380× | 1.839× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6180 ns | 5250 ns | -930 ns (-15.0%) | 3360 ns | -2820 ns (-45.6%) |
| SM Throughput (%) | 16.35 % | 35.34 % | +18.99 pp | 16.03 % | -0.32 pp |
| Memory Throughput (%) | 39.99 % | 36.18 % | -3.81 pp | 20.44 % | -19.55 pp |
| DRAM Throughput (%) | 20.03 % | 12.61 % | -7.42 pp | 4.95 % | -15.08 pp |
| Achieved Occupancy (%) | 54.73 % | 74.56 % | +19.83 pp | 46.38 % | -8.35 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 74.39 % | 38.08 % | -36.31 pp | 55.16 % | -19.23 pp |
| IPC Active | 1.16 inst/cycle | 2.40 inst/cycle | +1.24 | 1.49 inst/cycle | +0.33 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_max_contiguous_multiple_of` (1.177×)

When a Triton kernel's dominant bottleneck is L1TEX scoreboard stalls (CPIStall rule, ~48% of stall cycles) combined with very low eligible-warp rate (<1 warp/scheduler) and high no-eligible-cycle percentage (>70%), the compiler is likely issuing scalar 32-bit loads instead of 128-bit vectorized loads. Wrapping the index computation with `tl.max_contiguous(tl.multiple_of(offset + tl.arange(0, BLOCK), BLOCK), BLOCK)` provides the compiler with alignment and contiguity guarantees needed to emit vectorized ld.global.v4 instructions. This reduced warp cycles per issued instruction from 30.2 to 19.6, doubled issue slot utilization (29% → 61%), and improved achieved occupancy from 54.7% to 74.6%, yielding a 1.18x speedup.

> **Applicable when**: CPIStall rule triggered with L1TEX stall accounting for >40% of stall cycles; no-eligible-warp cycles >60%; eligible warps per scheduler <1.5; the load index in the innermost (x) dimension is provably stride-1 contiguous (i.e., xindex = xoffset + arange(0, XBLOCK) with xoffset = program_id * XBLOCK); waves_per_sm < 1.0 indicating underutilization

> **Avoid when**: The access pattern is not stride-1 contiguous in the indexed dimension (e.g., strided, indirect, or gather accesses); the base pointer is not guaranteed to be XBLOCK-aligned; the kernel is already compute-bound rather than memory-latency-bound

---

## Cumulative Progress

```
Baseline:  6180 ns  (1.000×)
Iter 1     3360 ns  (1.839×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 3360 ns  (1.8393×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=4),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +40,13 @@

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
