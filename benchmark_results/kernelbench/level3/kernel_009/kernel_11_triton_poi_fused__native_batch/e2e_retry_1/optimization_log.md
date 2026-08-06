# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3710 ns — **Best latency**: 3520 ns (**1.0540×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_increase_tile_size`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 22
- **Duration**: 3710 ns
- **SM Throughput**: 16.97 % | **Memory**: 8.94 % | **DRAM**: 8.94 %
- **Achieved Occupancy**: 56.24 %
- **IPC Active**: 1.44 inst/cycle | **No Eligible**: 56.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.005× | 1.054× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.956× | 0.905× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3520 ns | -190 ns (-5.1%) | 4100 ns | +390 ns (+10.5%) |
| SM Throughput (%) | 16.97 % | 25.64 % | +8.67 pp | 34.47 % | +17.50 pp |
| Memory Throughput (%) | 8.94 % | 9.86 % | +0.92 pp | 10.14 % | +1.20 pp |
| DRAM Throughput (%) | 8.94 % | 9.43 % | +0.49 pp | 8.12 % | -0.82 pp |
| Achieved Occupancy (%) | 56.24 % | 53.83 % | -2.41 pp | 52.79 % | -3.45 pp |
| Registers/Thread | 22.00 register/thread | 28.00 register/thread | +6 | 32.00 register/thread | +10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 56.05 % | 37.15 % | -18.90 pp | 28.97 % | -27.08 pp |
| IPC Active | 1.44 inst/cycle | 2.42 inst/cycle | +0.98 | 2.76 inst/cycle | +1.32 |


### Key Learnings

#### ✅ Plan A: `autotune_xblock_increase_tile_size` (1.054×)

When a pointwise Triton kernel has low eligible warps per scheduler (<2) and high 'No Eligible' percentage (>50%), the per-block work is too small to hide instruction latency. Replacing a fixed small XBLOCK (e.g. 128) with @triton.autotune over larger tile sizes (256, 512, 1024) allows each block to issue more instructions per launch, raising Executed IPC Active (1.44→2.42) and reducing Warp Cycles Per Issued Instruction (20.3→13.9). The speedup is modest (~5%) when the grid is already wave-limited (0.74 waves/SM), because the bottleneck shifts from scheduler stalls to the fixed wave count rather than achieving the 40-80% predicted; however, compute throughput improved from 17% to 26% and eligible warps nearly doubled.

> **Applicable when**: Pointwise Triton kernel with: eligible warps per scheduler < 2, No Eligible % > 40%, Warp Cycles Per Issued Instruction > 15, Compute (SM) Throughput < 30%, and the bottleneck is scheduler stalls (CPIStall on L1TEX). Grid is small relative to SM count (waves_per_sm < 1.5). XBLOCK is hardcoded to a small value (≤128).

> **Avoid when**: Kernel is already memory-bandwidth bound (DRAM Throughput > 70%). Increasing XBLOCK would exceed L1 cache capacity for data that must be reused within a block. Grid is already large (waves_per_sm > 2) and occupancy is near theoretical. The xnumel is not evenly divisible by the candidate XBLOCK values (would require careful masking). Autotune overhead is unacceptable in latency-critical single-invocation contexts.

---

## Cumulative Progress

```
Baseline:  3710 ns  (1.000×)
Iter 1     3520 ns  (1.054×)  — autotune_xblock_increase_tile_size

Final best: 3520 ns  (1.0540×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 200704
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x2), xmask)
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -33,9 +45,7 @@

     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1], 0, tl.int32)
     tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +56,7 @@

     arg_5 = inputs[5]
     xnumel = 200704
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 200704, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 200704)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
