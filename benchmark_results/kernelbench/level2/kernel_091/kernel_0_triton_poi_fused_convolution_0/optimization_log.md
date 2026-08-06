# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 4160 ns (**1.4231×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_and_evict_first_strided_store`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_091/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 15.86 % | **Memory**: 41.17 % | **DRAM**: 20.79 %
- **Achieved Occupancy**: 54.82 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.354× | 1.762× | ✓ |  |
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.399× | 1.423× | ✓ | **✓ best** |
| C | memory_coalescing_store_layout | 1.368× | 1.885× | ✓ |  |
| D | autotune_tile_sizes_and_warps_for_underutilized_layout_transpose | 1.275× | 1.469× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3360 ns | -2560 ns (-43.2%) | 4160 ns | -1760 ns (-29.7%) | 3140 ns | -2780 ns (-47.0%) | 4030 ns | -1890 ns (-31.9%) |
| SM Throughput (%) | 15.86 % | 17.21 % | +1.35 pp | 26.94 % | +11.08 pp | 15.49 % | -0.37 pp | 19.22 % | +3.36 pp |
| Memory Throughput (%) | 41.17 % | 20.50 % | -20.67 pp | 21.45 % | -19.72 pp | 21.91 % | -19.26 pp | 22.23 % | -18.94 pp |
| DRAM Throughput (%) | 20.79 % | 4.95 % | -15.84 pp | 3.98 % | -16.81 pp | 5.28 % | -15.51 pp | 4.10 % | -16.69 pp |
| Achieved Occupancy (%) | 54.82 % | 46.20 % | -8.62 pp | 67.52 % | +12.70 pp | 46.00 % | -8.82 pp | 48.15 % | -6.67 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 16.00 register/thread | -10 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.42 % | 55.27 % | -15.15 pp | 38.13 % | -32.29 pp | 61.32 % | -9.10 pp | 51.84 % | -18.58 pp |
| IPC Active | 1.15 inst/cycle | 1.57 inst/cycle | +0.42 | 2.38 inst/cycle | +1.23 | 1.47 inst/cycle | +0.32 | 1.82 inst/cycle | +0.67 |

---

## Iteration 2

**Best so far**: 1.4231× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_max_contiguous_multiple_of_hints | 1.359× | 1.867× | ✓ |  |
| B | shared_memory_tiling_for_coalesced_store | 1.363× | 1.794× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints | 1.300× | 1.423× | ✓ |  |
| D | vectorized_loads_with_contiguous_hints | 1.369× | 1.502× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_til... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3170 ns | -2750 ns (-46.5%) | 3300 ns | -2620 ns (-44.3%) | 4160 ns | -1760 ns (-29.7%) | 3940 ns | -1980 ns (-33.4%) |
| SM Throughput (%) | 15.86 % | 15.31 % | -0.55 pp | 17.92 % | +2.06 pp | 18.84 % | +2.98 pp | 25.78 % | +9.92 pp |
| Memory Throughput (%) | 41.17 % | 21.77 % | -19.40 pp | 20.74 % | -20.43 pp | 22.32 % | -18.85 pp | 20.51 % | -20.66 pp |
| DRAM Throughput (%) | 20.79 % | 5.24 % | -15.55 pp | 5.03 % | -15.76 pp | 3.97 % | -16.82 pp | 4.22 % | -16.57 pp |
| Achieved Occupancy (%) | 54.82 % | 45.38 % | -9.44 pp | 45.69 % | -9.13 pp | 49.62 % | -5.20 pp | 68.58 % | +13.76 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.42 % | 64.96 % | -5.46 pp | 54.45 % | -15.97 pp | 52.73 % | -17.69 pp | 37.98 % | -32.44 pp |
| IPC Active | 1.15 inst/cycle | 1.42 inst/cycle | +0.27 | 1.66 inst/cycle | +0.51 | 1.73 inst/cycle | +0.58 | 2.43 inst/cycle | +1.28 |

---

## Iteration 3

**Best so far**: 1.4231× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_store_hint_for_strided_scatter_writes | 1.300× | 1.885× | ✓ |  |
| B | evict_first_store_with_expanded_autotune_configs | 1.335× | 1.778× | ✓ |  |
| C | fix_autotune_grid_lambda_and_evict_first_strided_store | 1.383× | 1.423× | ✓ |  |
| D | increase_num_stages_to_pipeline_memory_latency | 1.252× | 1.033× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_store... | Δ | evict_first_store... | Δ | fix_autotune_grid... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3140 ns | -2780 ns (-47.0%) | 3330 ns | -2590 ns (-43.8%) | 4160 ns | -1760 ns (-29.7%) | 5730 ns | -190 ns (-3.2%) |
| SM Throughput (%) | 15.86 % | 16.37 % | +0.51 pp | 17.59 % | +1.73 pp | 27.86 % | +12.00 pp | 27.02 % | +11.16 pp |
| Memory Throughput (%) | 41.17 % | 21.87 % | -19.30 pp | 20.46 % | -20.71 pp | 20.94 % | -20.23 pp | 29.58 % | -11.59 pp |
| DRAM Throughput (%) | 20.79 % | 5.30 % | -15.49 pp | 4.95 % | -15.84 pp | 3.98 % | -16.81 pp | 2.87 % | -17.92 pp |
| Achieved Occupancy (%) | 54.82 % | 46.05 % | -8.77 pp | 52.40 % | -2.42 pp | 71.84 % | +17.02 pp | 73.34 % | +18.52 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.42 % | 57.33 % | -13.09 pp | 54.49 % | -15.93 pp | 39.14 % | -31.28 pp | 38.05 % | -32.37 pp |
| IPC Active | 1.15 inst/cycle | 1.52 inst/cycle | +0.37 | 1.35 inst/cycle | +0.20 | 2.26 inst/cycle | +1.11 | 2.41 inst/cycle | +1.26 |


### Key Learnings

#### ✅ Plan A: `evict_first_store_hint_for_strided_scatter_writes` (1.885×)

For Triton kernels with confirmed uncoalesced global stores (NCU UncoalescedGlobalAccess, ~50% excessive sectors, ~10.7/32 bytes utilized per sector) performing stride-C scatter writes (e.g., NCHW→NHWC layout transpose), adding `eviction_policy='evict_first'` to the strided tl.store call reduces L2 cache pollution from non-reused dirty write lines. This improved scheduler efficiency: no-eligible cycles dropped from 64.96% to 57.33% and eligible warps per scheduler rose from 1.33 to 1.63, while the uncoalesced sector count remained unchanged (50176), confirming evict_first reduces cache thrash rather than fixing the access pattern. The wall-clock improvement in this iteration was modest (~1% duration reduction from 3.17µs to 3.14µs), with the overall 1.89x speedup attributable primarily to prior autotune and vectorization changes already applied.

> **Applicable when**: Triton kernel has confirmed uncoalesced global stores (NCU UncoalescedGlobalAccess rule, >40% excessive sectors, <12 bytes/sector utilized); the store access pattern is a stride-C scatter (e.g., layout transpose); IssueSlotUtilization shows >50% no-eligible-warp cycles; kernel is already autotuned; L2 hit rate on reads is high (>70%), indicating L2 is being used for read caching that write pollution could displace.

> **Avoid when**: The store pattern is already coalesced (evict_first provides no benefit); the kernel is compute-bound rather than memory-latency-bound; the bottleneck is DRAM bandwidth rather than L2 cache thrash; the wall-clock improvement is the primary goal and prior iterations show this hint yields only marginal duration gains (~1%) even when scheduler metrics improve.

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     4160 ns  (1.423×)  — autotune_tile_sizes_fix_uncoalesced_stores

Final best: 4160 ns  (1.4231×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8,  num_stages=3),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
+def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
     ynumel = 30
     xnumel = 50176
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Vectorization hints: x-dim is stride-1 contiguous and XBLOCK-aligned
+    xindex_1d = xoffset + tl.arange(0, XBLOCK)
+    xindex_1d = tl.max_contiguous(tl.multiple_of(xindex_1d, XBLOCK), XBLOCK)
+    xindex = xindex_1d[None, :]  # restore 2D shape [1, XBLOCK]
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

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
