# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3170 ns (**1.8770×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`shared_memory_transpose_to_coalesce_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_011/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.51 % | **Memory**: 41.66 % | **DRAM**: 20.69 %
- **Achieved Occupancy**: 54.87 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.274× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_for_layout_transpose_kernel | 1.443× | 1.877× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_reduce_grid_fragmentation_and_l1tex_stalls | 1.022× | 0.954× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 5950 ns | +0 ns (+0.0%) | 3170 ns | -2780 ns (-46.7%) | 6240 ns | +290 ns (+4.9%) |
| SM Throughput (%) | 16.51 % | 16.31 % | -0.20 pp | 5.08 % | -11.43 pp | 16.65 % | +0.14 pp |
| Memory Throughput (%) | 41.66 % | 41.32 % | -0.34 pp | 11.27 % | -30.39 pp | 39.65 % | -2.01 pp |
| DRAM Throughput (%) | 20.69 % | 20.67 % | -0.02 pp | 2.63 % | -18.06 pp | 19.77 % | -0.92 pp |
| Achieved Occupancy (%) | 54.87 % | 55.24 % | +0.37 pp | 26.96 % | -27.91 pp | 55.07 % | +0.20 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.58 % | 70.49 % | -0.09 pp | 79.13 % | +8.55 pp | 71.34 % | +0.76 pp |
| IPC Active | 1.15 inst/cycle | 1.15 inst/cycle | +0.00 | 0.70 inst/cycle | -0.45 | 1.15 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.877× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.411× | 1.859× | ✓ |  |
| B | add_yblock1_configs_and_eliminate_dynamic_smem | 1.434× | 1.328× | ✓ |  |
| C | add_yblock1_configs_with_pipelining_to_autotune | 1.430× | 1.859× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | add_yblock1_confi... | Δ | add_yblock1_confi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3200 ns | -2750 ns (-46.2%) | 4480 ns | -1470 ns (-24.7%) | 3200 ns | -2750 ns (-46.2%) |
| SM Throughput (%) | 16.51 % | 5.10 % | -11.41 pp | 25.12 % | +8.61 pp | 4.97 % | -11.54 pp |
| Memory Throughput (%) | 41.66 % | 11.32 % | -30.34 pp | 19.99 % | -21.67 pp | 11.38 % | -30.28 pp |
| DRAM Throughput (%) | 20.69 % | 2.60 % | -18.09 pp | 3.69 % | -17.00 pp | 2.61 % | -18.08 pp |
| Achieved Occupancy (%) | 54.87 % | 26.02 % | -28.85 pp | 68.65 % | +13.78 pp | 26.79 % | -28.08 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.58 % | 79.76 % | +9.18 pp | 40.09 % | -30.49 pp | 79.42 % | +8.84 pp |
| IPC Active | 1.15 inst/cycle | 0.65 inst/cycle | -0.50 | 2.30 inst/cycle | +1.15 | 0.60 inst/cycle | -0.55 |

---

## Iteration 3

**Best so far**: 1.877× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_transpose_to_coalesce_stores | 1.412× | 1.877× | ✓ |  |
| B | evict_first_store_policy_for_strided_writes | 1.440× | 1.877× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ | evict_first_store... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3170 ns | -2780 ns (-46.7%) | 3170 ns | -2780 ns (-46.7%) |
| SM Throughput (%) | 16.51 % | 5.32 % | -11.19 pp | 5.68 % | -10.83 pp |
| Memory Throughput (%) | 41.66 % | 11.33 % | -30.33 pp | 11.23 % | -30.43 pp |
| DRAM Throughput (%) | 20.69 % | 2.62 % | -18.07 pp | 2.62 % | -18.07 pp |
| Achieved Occupancy (%) | 54.87 % | 24.93 % | -29.94 pp | 27.00 % | -27.87 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 70.58 % | 77.91 % | +7.33 pp | 79.42 % | +8.84 pp |
| IPC Active | 1.15 inst/cycle | 0.63 inst/cycle | -0.52 | 0.79 inst/cycle | -0.36 |


### Key Learnings

#### ✅ Plan B: `evict_first_store_policy_for_strided_writes` (1.877×)

For kernels with uncoalesced stores (e.g. stride-3 writes where only 10.7 of 32 bytes per sector are utilized), adding eviction_policy='evict_first' to tl.store marks writes as streaming/non-temporal, reducing L1 cache pollution from data that will never be re-read by the kernel. This improved Warp Cycles Per Issued Instruction from 22.1 to 19.92 and Executed IPC Active from 0.65 to 0.79, indicating better instruction throughput. However, the fundamental uncoalesced access pattern (50% excessive sectors) remains unchanged, so the improvement is modest — the cache policy hint reduces contention between strided stores and contiguous loads but does not eliminate the underlying bandwidth waste.

> **Applicable when**: Triton kernel with uncoalesced global stores (MemoryCacheAccessPattern rule triggered, bytes-per-sector < 16 of 32); store data is never re-read by the same kernel (write-once pattern); CPIStall triggered by L1TEX scoreboard stalls; contiguous loads coexist with strided stores in the same kernel (load path benefits from freed L1 capacity); UncoalescedGlobalAccess rule shows 40%+ excessive sectors.

> **Avoid when**: Store data is re-read within the same kernel (evict_first would cause cache misses on subsequent reads); the store pattern is already coalesced (no benefit from streaming hint); the kernel is purely store-bound with no competing loads (no L1 contention to relieve); NCU elapsed cycles show no improvement despite the hint (indicates the bottleneck is elsewhere, e.g. grid occupancy or compute).

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3170 ns  (1.877×)  — autotune_tile_sizes_for_layout_transpose_kernel

Final best: 3170 ns  (1.8770×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 4},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -14,6 +34,10 @@

     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Vectorization hints: xoffset is always a multiple of XBLOCK (power-of-2),
+    # and XBLOCK consecutive xindex values are contiguous. This enables LDG.128.
+    # xindex has shape [1, XBLOCK], so values must be a list [1, XBLOCK].
+    xindex = tl.max_contiguous(tl.multiple_of(xindex, [1, XBLOCK]), [1, XBLOCK])
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +46,13 @@

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
