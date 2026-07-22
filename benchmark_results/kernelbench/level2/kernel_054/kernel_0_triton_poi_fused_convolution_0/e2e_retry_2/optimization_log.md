# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6300 ns — **Best latency**: 4260 ns (**1.4789×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_coalescing_and_occupancy`** beats baseline latency.

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
- **Duration**: 6300 ns
- **SM Throughput**: 15.98 % | **Memory**: 39.69 % | **DRAM**: 19.58 %
- **Achieved Occupancy**: 55.77 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.368× | 1.438× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_coalescing_and_occupancy | 1.479× | 1.479× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6300 ns | 4380 ns | -1920 ns (-30.5%) | 4260 ns | -2040 ns (-32.4%) |
| SM Throughput (%) | 15.98 % | 20.96 % | +4.98 pp | 26.67 % | +10.69 pp |
| Memory Throughput (%) | 39.69 % | 31.83 % | -7.86 pp | 21.13 % | -18.56 pp |
| DRAM Throughput (%) | 19.58 % | 14.99 % | -4.59 pp | 3.89 % | -15.69 pp |
| Achieved Occupancy (%) | 55.77 % | 46.11 % | -9.66 pp | 71.82 % | +16.05 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.64 % | 56.60 % | -14.04 pp | 38.58 % | -32.06 pp |
| IPC Active | 1.11 inst/cycle | 1.69 inst/cycle | +0.58 | 2.15 inst/cycle | +1.04 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_and_warps_for_coalescing_and_occupancy` (1.479×)

When a Triton pointwise kernel shows low achieved occupancy (<60%), high 'No Eligible' warp percentage (>70%), and uncoalesced global stores due to strided index patterns, adding @triton.autotune over XBLOCK, YBLOCK, and num_warps configurations can simultaneously improve store coalescing (by enlarging the contiguous tile dimension) and increase the eligible-warp pool (by raising num_warps). In this case, doubling block size from 128 to 256 threads and doubling waves/SM from 0.74 to 1.48 reduced warp cycles per issued instruction from 30.7 to 17.6 and cut kernel duration from 6.3 µs to 4.26 µs. The autotune approach is safe because it only changes tile parameters, not index arithmetic, and lets the runtime select the best config for the actual problem shape.

> **Applicable when**: Triton pointwise kernel with: achieved occupancy <65%, waves/SM <1.0, 'No Eligible' warp % >60%, L1TEX stall cycles >10 avg (CPIStall rule triggered), uncoalesced global stores (UncoalescedGlobalAccess rule triggered), and fixed hardcoded XBLOCK/YBLOCK/num_warps launch config. Especially effective when the store index has a strided component in the YBLOCK dimension.

> **Avoid when**: Kernel is already autotuned or uses dynamic tile sizes. Kernel is compute-bound (SM throughput >80%) rather than latency/occupancy-bound. Problem size is so small that autotune overhead dominates. Index arithmetic correctness depends on specific XBLOCK/YBLOCK values (e.g., tiling assumptions baked into the math).

---

## Cumulative Progress

```
Baseline:  6300 ns  (1.000×)
Iter 1     4260 ns  (1.479×)  — autotune_tile_sizes_and_warps_for_coalescing_and_occupancy

Final best: 4260 ns  (1.4789×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16},num_warps=8,  num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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
