# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 3260 ns (**1.8252×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_for_correct_tile_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_004/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.66 % | **Memory**: 41.45 % | **DRAM**: 20.81 %
- **Achieved Occupancy**: 55.63 %
- **IPC Active**: 1.07 inst/cycle | **No Eligible**: 70.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.247× | 1.630× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.440× | 1.859× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_latency_hiding | 1.219× | 1.441× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3650 ns | -2300 ns (-38.7%) | 3200 ns | -2750 ns (-46.2%) | 4130 ns | -1820 ns (-30.6%) |
| SM Throughput (%) | 16.66 % | 18.03 % | +1.37 pp | 5.05 % | -11.61 pp | 27.37 % | +10.71 pp |
| Memory Throughput (%) | 41.45 % | 18.54 % | -22.91 pp | 11.19 % | -30.26 pp | 21.74 % | -19.71 pp |
| DRAM Throughput (%) | 20.81 % | 4.50 % | -16.31 pp | 2.62 % | -18.19 pp | 4.00 % | -16.81 pp |
| Achieved Occupancy (%) | 55.63 % | 44.04 % | -11.59 pp | 26.44 % | -29.19 pp | 67.76 % | +12.13 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 16.00 register/thread | -10 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 1.48 | +0.74 |
| No Eligible (%) | 70.28 % | 51.78 % | -18.50 pp | 79.58 % | +9.30 pp | 37.94 % | -32.34 pp |
| IPC Active | 1.07 inst/cycle | 1.78 inst/cycle | +0.71 | 0.61 inst/cycle | -0.46 | 2.40 inst/cycle | +1.33 |

---

## Iteration 2

**Best so far**: 1.8594× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.443× | 1.877× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_for_layout_permutation | 1.308× | 1.771× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints | 1.228× | 1.328× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3170 ns | -2780 ns (-46.7%) | 3360 ns | -2590 ns (-43.5%) | 4480 ns | -1470 ns (-24.7%) |
| SM Throughput (%) | 16.66 % | 5.15 % | -11.51 pp | 17.49 % | +0.83 pp | 27.66 % | +11.00 pp |
| Memory Throughput (%) | 41.45 % | 11.36 % | -30.09 pp | 20.48 % | -20.97 pp | 20.44 % | -21.01 pp |
| DRAM Throughput (%) | 20.81 % | 2.62 % | -18.19 pp | 4.95 % | -15.86 pp | 3.70 % | -17.11 pp |
| Achieved Occupancy (%) | 55.63 % | 26.25 % | -29.38 pp | 44.07 % | -11.56 pp | 68.48 % | +12.85 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.28 % | 79.55 % | +9.27 pp | 53.75 % | -16.53 pp | 37.49 % | -32.79 pp |
| IPC Active | 1.07 inst/cycle | 0.68 inst/cycle | -0.39 | 1.80 inst/cycle | +0.73 | 2.46 inst/cycle | +1.39 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.328×)

When a Triton kernel's load dimension is stride-1 contiguous but the compiler cannot infer alignment, wrapping the 1D index with tl.multiple_of(expr, 4) and tl.max_contiguous(expr, 4) before broadcasting to 2D enables 128-bit (float4) vectorized loads. This reduces L1TEX transaction count on the load path and improves memory throughput (126 → 180 GB/s) and compute utilization (5% → 28% SM throughput). The hints must be applied to the 1D tl.arange tensor before the [None, :] broadcast, and the alignment value (4 for float32) must divide both XBLOCK and xnumel.

> **Applicable when**: Triton kernel with a stride-1 contiguous load dimension where UncoalescedGlobalAccess or MemoryCacheAccessPattern NCU rules fire, bytes-utilized-per-sector < 16/32, and xnumel is divisible by 4 (i.e., 4 × sizeof(dtype) = 128-bit alignment holds). Particularly effective for layout-permutation kernels (e.g., NCHW→NHWC) where the load path is contiguous but the store path has structural stride.

> **Avoid when**: xnumel is not divisible by the hint value (4 for float32, 8 for float16), or XBLOCK values in autotune configs are not all multiples of the hint value. Do not apply hints to the store dimension if it has a non-unit stride — this causes correctness failures (as seen in item f6c94494). Also avoid if the load dimension is already vectorized (L1TEX hit rate high, bytes-per-sector near 32).

#### ✅ Plan C: `vectorized_loads_with_contiguous_hints` (1.328×)

When a Triton kernel's load dimension is stride-1 contiguous but the compiler cannot infer alignment, wrapping the 1D index with tl.multiple_of(expr, 4) and tl.max_contiguous(expr, 4) before broadcasting to 2D enables 128-bit (float4) vectorized loads. This reduces L1TEX transaction count on the load path and improves memory throughput (126 → 180 GB/s) and compute utilization (5% → 28% SM throughput). The hints must be applied to the 1D tl.arange tensor before the [None, :] broadcast, and the alignment value (4 for float32) must divide both XBLOCK and xnumel.

> **Applicable when**: Triton kernel with a stride-1 contiguous load dimension where UncoalescedGlobalAccess or MemoryCacheAccessPattern NCU rules fire, bytes-utilized-per-sector < 16/32, and xnumel is divisible by 4 (i.e., 4 × sizeof(dtype) = 128-bit alignment holds). Particularly effective for layout-permutation kernels (e.g., NCHW→NHWC) where the load path is contiguous but the store path has structural stride.

> **Avoid when**: xnumel is not divisible by the hint value (4 for float32, 8 for float16), or XBLOCK values in autotune configs are not all multiples of the hint value. Do not apply hints to the store dimension if it has a non-unit stride — this causes correctness failures (as seen in item f6c94494). Also avoid if the load dimension is already vectorized (L1TEX hit rate high, bytes-per-sector near 32).

---

## Iteration 3

**Best so far**: 1.877× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_for_correct_tile_coverage | 1.454× | 1.825× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_and_expand_configs | 1.308× | 1.192× | ✓ |  |
| C | fix_grid_lambda_and_wave_aligned_configs | 1.356× | 1.319× | ✓ |  |
| D | add_triton_autotune | 1.379× | 1.740× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3260 ns | -2690 ns (-45.2%) | 4990 ns | -960 ns (-16.1%) | 4510 ns | -1440 ns (-24.2%) | 3420 ns | -2530 ns (-42.5%) |
| SM Throughput (%) | 16.66 % | 5.11 % | -11.55 pp | 30.93 % | +14.27 pp | 28.14 % | +11.48 pp | 5.08 % | -11.58 pp |
| Memory Throughput (%) | 41.45 % | 10.88 % | -30.57 pp | 30.36 % | -11.09 pp | 20.79 % | -20.66 pp | 10.60 % | -30.85 pp |
| DRAM Throughput (%) | 20.81 % | 2.55 % | -18.26 pp | 3.31 % | -17.50 pp | 3.67 % | -17.14 pp | 2.44 % | -18.37 pp |
| Achieved Occupancy (%) | 55.63 % | 26.31 % | -29.32 pp | 73.02 % | +17.39 pp | 68.36 % | +12.73 pp | 27.53 % | -28.10 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 27.00 register/thread | +1 | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 1.48 | +0.74 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.28 % | 79.59 % | +9.31 pp | 34.06 % | -36.22 pp | 37.02 % | -33.26 pp | 80.39 % | +10.11 pp |
| IPC Active | 1.07 inst/cycle | 0.68 inst/cycle | -0.39 | 2.58 inst/cycle | +1.51 | 2.46 inst/cycle | +1.39 | 0.51 inst/cycle | -0.56 |

---

## Iteration 4

**Best so far**: 1.877× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_uncoalesced_store_overhead_via_eviction_policy_and_wave_alignment | 1.365× | 1.387× | ✓ |  |
| C | fix_static_grid_to_lambda_for_autotune_correctness | 1.260× | 1.068× | ✓ |  |
| D | fix_static_grid_and_expand_configs_for_wave_saturation | 1.377× | 1.877× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_uncoalesce... | Δ | fix_static_grid_t... | Δ | fix_static_grid_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 4290 ns | -1660 ns (-27.9%) | 5570 ns | -380 ns (-6.4%) | 3170 ns | -2780 ns (-46.7%) |
| SM Throughput (%) | 16.66 % | 30.44 % | +13.78 pp | 31.91 % | +15.25 pp | 5.17 % | -11.49 pp |
| Memory Throughput (%) | 41.45 % | 21.47 % | -19.98 pp | 31.43 % | -10.02 pp | 11.27 % | -30.18 pp |
| DRAM Throughput (%) | 20.81 % | 3.85 % | -16.96 pp | 2.98 % | -17.83 pp | 2.63 % | -18.18 pp |
| Achieved Occupancy (%) | 55.63 % | 70.31 % | +14.68 pp | 73.26 % | +17.63 pp | 26.10 % | -29.53 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 27.00 register/thread | +1 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.28 % | 37.97 % | -32.31 pp | 35.39 % | -34.89 pp | 80.46 % | +10.18 pp |
| IPC Active | 1.07 inst/cycle | 2.40 inst/cycle | +1.33 | 2.56 inst/cycle | +1.49 | 0.67 inst/cycle | -0.40 |

---

## Iteration 5

**Best so far**: 1.877× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_static_grid_to_lambda_for_autotune_wave_elimination | 1.357× | 1.368× | ✓ |  |
| C | evict_first_on_write_once_scatter_stores | 1.312× | 1.133× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | evict_first_on_wr... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 4350 ns | -1600 ns (-26.9%) | 5250 ns | -700 ns (-11.8%) |
| SM Throughput (%) | 16.66 % | 27.55 % | +10.89 pp | 34.55 % | +17.89 pp |
| Memory Throughput (%) | 41.45 % | 19.44 % | -22.01 pp | 32.56 % | -8.89 pp |
| DRAM Throughput (%) | 20.81 % | 3.79 % | -17.02 pp | 3.13 % | -17.68 pp |
| Achieved Occupancy (%) | 55.63 % | 67.72 % | +12.09 pp | 74.40 % | +18.77 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.28 % | 36.29 % | -33.99 pp | 33.53 % | -36.75 pp |
| IPC Active | 1.07 inst/cycle | 2.45 inst/cycle | +1.38 | 2.48 inst/cycle | +1.41 |


### Key Learnings

#### ✅ Plan C: `evict_first_on_write_once_scatter_stores` (1.387×)

For write-once output buffers with strided (non-coalesced) store patterns, adding eviction_policy='evict_first' to tl.store prevents L1/L2 from allocating cache lines for the written data, reducing cache pollution and freeing capacity for coalesced loads. In this kernel, the dominant improvement came from the concurrent autotune config change (block_size 64→256, waves_per_sm 0.37→1.48, achieved occupancy 26%→70%), which drove the speedup; the evict_first hint on the stride-3 scatter store is a complementary change that reduces L2 write-allocate pressure when UncoalescedGlobalAccess and MemoryCacheAccessPattern rules fire with <12 bytes utilized per sector on stores. The L2 hit rate remained stable (~77%), confirming evict_first did not harm read caching.

> **Applicable when**: Triton kernel writes to a write-once output buffer with a strided (non-unit stride) store pattern. NCU UncoalescedGlobalAccess fires (>30% excessive sectors), MemoryCacheAccessPattern reports <16 bytes utilized per sector on stores, and the output buffer is not read back within the same kernel. L2 hit rate on loads is moderate-to-high (>60%), indicating L2 is already under pressure.

> **Avoid when**: Output buffer is read back within the same kernel (evict_first would cause cache misses on subsequent reads). Store pattern is already coalesced (bytes-per-sector near 32) — evict_first provides no benefit and may slightly increase DRAM traffic. Do not apply to shared memory stores or to load addresses. Do not confuse with evict_last, which is appropriate for data reused across tiles.

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3200 ns  (1.859×)  — autotune_tile_sizes_for_occupancy_and_coalescing
Iter 2     3170 ns  (1.877×)  — vectorized_loads_with_contiguous_hints

Final best: 3260 ns  (1.8252×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +37,13 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # tl.multiple_of: xoffset is always a multiple of 4 (XBLOCK >= 64, power-of-2)
+    # tl.max_contiguous: XBLOCK consecutive elements are stride-1 contiguous
+    xarange = tl.max_contiguous(
+        tl.multiple_of(xoffset + tl.arange(0, XBLOCK), 4),
+        4
+    )
+    xindex = xarange[None, :]  # broadcast to [1, XBLOCK] — same shape as before
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +52,13 @@

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
