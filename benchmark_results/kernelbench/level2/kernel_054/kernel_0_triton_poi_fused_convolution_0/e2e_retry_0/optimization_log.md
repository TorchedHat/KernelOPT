# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6340 ns — **Best latency**: 4290 ns (**1.4779×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_054/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6340 ns
- **SM Throughput**: 16.13 % | **Memory**: 40.00 % | **DRAM**: 19.54 %
- **Achieved Occupancy**: 55.29 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.375× | 1.478× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.053× | 1.065× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6340 ns | 4290 ns | -2050 ns (-32.3%) | 5950 ns | -390 ns (-6.2%) |
| SM Throughput (%) | 16.13 % | 17.00 % | +0.87 pp | 16.41 % | +0.28 pp |
| Memory Throughput (%) | 40.00 % | 32.19 % | -7.81 pp | 41.60 % | +1.60 pp |
| DRAM Throughput (%) | 19.54 % | 15.42 % | -4.12 pp | 20.72 % | +1.18 pp |
| Achieved Occupancy (%) | 55.29 % | 53.60 % | -1.69 pp | 55.39 % | +0.10 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.62 % | 65.00 % | -5.62 pp | 71.65 % | +1.03 pp |
| IPC Active | 1.11 inst/cycle | 1.34 inst/cycle | +0.23 | 1.11 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.478×)

When a Triton pointwise kernel has uncoalesced global stores (UncoalescedGlobalAccess rule, >15% excessive sectors) and low SM utilization (<1 wave), replacing fixed XBLOCK/YBLOCK constants with @triton.autotune over a grid of (XBLOCK, YBLOCK, num_warps, num_stages) configurations lets the runtime select the tile shape that minimizes store stride amplification and maximizes CTA count. The winning config reduced warp cycles per issued instruction from 34 to 24 and L2 sector traffic by ~42%, yielding a 1.48x speedup. Register pressure also dropped (26→21 registers/thread) as a side effect of the autotuner selecting a lower-register config.

> **Applicable when**: Triton pointwise kernel with: (1) UncoalescedGlobalAccess rule triggered (>15% excessive sectors), (2) waves_per_sm < 1.0 indicating SM underutilization, (3) Warp Cycles Per Issued Instruction > 25 with CPIStall pointing to L1TEX latency, (4) fixed XBLOCK/YBLOCK tile sizes hardcoded in the launcher. Especially effective for layout-permutation kernels (e.g., NCHW→NHWC) where store stride is non-unit.

> **Avoid when**: Kernel already uses autotune; kernel has waves_per_sm >> 1 (SM already saturated); kernel is compute-bound rather than memory-latency-bound; autotune overhead is unacceptable in latency-critical single-invocation paths.

---

## Cumulative Progress

```
Baseline:  6340 ns  (1.000×)
Iter 1     4290 ns  (1.478×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 4290 ns  (1.4779×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK':  64, 'YBLOCK':  8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  32, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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
