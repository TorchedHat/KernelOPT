# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 5890 ns (**1.0221×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_lambda_for_autotune_correctness_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_017/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.90 % | **Memory**: 40.55 % | **DRAM**: 20.60 %
- **Achieved Occupancy**: 54.99 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.331× | 1.494× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.248× | 1.034× | ✓ |  |
| D | add_autotune_increase_tile_sizes | 1.081× | 0.436× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4030 ns | -1990 ns (-33.1%) | 5820 ns | -200 ns (-3.3%) | 13820 ns | +7800 ns (+129.6%) |
| SM Throughput (%) | 16.90 % | 27.51 % | +10.61 pp | 16.79 % | -0.11 pp | 48.67 % | +31.77 pp |
| Memory Throughput (%) | 40.55 % | 21.86 % | -18.69 pp | 41.49 % | +0.94 pp | 60.09 % | +19.54 pp |
| DRAM Throughput (%) | 20.60 % | 4.10 % | -16.50 pp | 21.07 % | +0.47 pp | 8.89 % | -11.71 pp |
| Achieved Occupancy (%) | 54.99 % | 67.84 % | +12.85 pp | 54.90 % | -0.09 pp | 63.94 % | +8.95 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 1.98 | +1.24 |
| No Eligible (%) | 70.93 % | 46.21 % | -24.72 pp | 70.60 % | -0.33 pp | 44.82 % | -26.11 pp |
| IPC Active | 1.13 inst/cycle | 2.37 inst/cycle | +1.24 | 1.14 inst/cycle | +0.01 | 2.21 inst/cycle | +1.08 |

---

## Iteration 2

**Best so far**: 1.4938× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_to_lambda_for_tail_wave_elimination | 1.317× | 1.413× | ✓ |  |
| B | fix_autotune_lambda_grid_and_add_vectorized_load_hints | 1.365× | 1.045× | ✓ | **✓ best** |
| C | fix_autotune_lambda_grid_and_add_vectorized_load_hints | 0.661× | 1.447× | ✓ |  |
| D | add_triton_autotune | 1.306× | 1.483× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_lamb... | Δ | fix_autotune_lamb... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4260 ns | -1760 ns (-29.2%) | 5760 ns | -260 ns (-4.3%) | 4160 ns | -1860 ns (-30.9%) | 4060 ns | -1960 ns (-32.6%) |
| SM Throughput (%) | 16.90 % | 26.23 % | +9.33 pp | 16.73 % | -0.17 pp | 26.16 % | +9.26 pp | 26.01 % | +9.11 pp |
| Memory Throughput (%) | 40.55 % | 20.84 % | -19.71 pp | 42.30 % | +1.75 pp | 20.78 % | -19.77 pp | 20.66 % | -19.89 pp |
| DRAM Throughput (%) | 20.60 % | 3.91 % | -16.69 pp | 21.38 % | +0.78 pp | 4.01 % | -16.59 pp | 4.09 % | -16.51 pp |
| Achieved Occupancy (%) | 54.99 % | 67.95 % | +12.96 pp | 54.89 % | -0.10 pp | 68.68 % | +13.69 pp | 71.43 % | +16.44 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.93 % | 36.86 % | -34.07 pp | 72.64 % | +1.71 pp | 38.66 % | -32.27 pp | 37.30 % | -33.63 pp |
| IPC Active | 1.13 inst/cycle | 2.39 inst/cycle | +1.26 | 1.16 inst/cycle | +0.03 | 2.39 inst/cycle | +1.26 | 2.27 inst/cycle | +1.14 |

---

## Iteration 3

**Best so far**: 1.4938× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | evict_first_streaming_store_for_strided_write | 1.377× | 1.393× | ✓ |  |
| C | fix_autotune_grid_lambda_and_larger_block_sizes | 1.406× | 1.045× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4320 ns | -1700 ns (-28.2%) | 5760 ns | -260 ns (-4.3%) |
| SM Throughput (%) | 16.90 % | 28.41 % | +11.51 pp | 16.78 % | -0.12 pp |
| Memory Throughput (%) | 40.55 % | 21.34 % | -19.21 pp | 42.23 % | +1.68 pp |
| DRAM Throughput (%) | 20.60 % | 3.83 % | -16.77 pp | 21.35 % | +0.75 pp |
| Achieved Occupancy (%) | 54.99 % | 68.69 % | +13.70 pp | 55.51 % | +0.52 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.93 % | 36.38 % | -34.55 pp | 72.85 % | +1.92 pp |
| IPC Active | 1.13 inst/cycle | 2.46 inst/cycle | +1.33 | 1.12 inst/cycle | -0.01 |


### Key Learnings

#### ✅ Plan B: `evict_first_streaming_store_for_strided_write` (1.393×)

Adding eviction_policy='evict_first' to tl.store for write-once strided output tensors is intended to free L2 capacity by immediately evicting store cache lines, but in this case the kernel duration actually increased (4.06 µs → 4.32 µs) despite a reported 1.39x end-to-end speedup. The L2 Hit Rate remained essentially unchanged (76.86% → 76.84%), indicating the eviction hint did not meaningfully reduce L2 pollution from the strided stores. The kernel was latency-bound (low DRAM throughput ~4%, low memory SOL ~20%) rather than L2-bandwidth-bound, so the eviction policy change had no positive effect on the binding bottleneck. The speedup observed is likely attributable to the simultaneous autotune config space expansion (additional configs added in the same diff) rather than the eviction_policy hint itself.

> **Applicable when**: Kernel is L2-bandwidth-bound (L2 throughput > 60%, DRAM throughput > 30%) AND stores are write-once strided data with high L2 hit rate on stores, indicating L2 is caching output data that is never re-read within the kernel.

> **Avoid when**: Kernel is latency-bound rather than bandwidth-bound (DRAM throughput < 10%, memory SOL < 30%); L2 hit rate is driven by input loads rather than output stores; the uncoalesced access pattern (excessive sectors) is the primary bottleneck rather than L2 capacity pressure. In these cases the eviction hint does not address the binding constraint and may slightly increase duration.

---

## Iteration 4

**Best so far**: 1.4938× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | flat_output_order_iteration_for_coalesced_stores | 1.335× | 1.229× | ✓ |  |
| B | shared_memory_staged_transpose_for_coalesced_store | 1.376× | 1.468× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_for_autotune_correctness_and_occupancy | 1.417× | 1.022× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | flat_output_order... | Δ | shared_memory_sta... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4900 ns | -1120 ns (-18.6%) | 4100 ns | -1920 ns (-31.9%) | 5890 ns | -130 ns (-2.2%) |
| SM Throughput (%) | 16.90 % | 25.07 % | +8.17 pp | 27.55 % | +10.65 pp | 17.23 % | +0.33 pp |
| Memory Throughput (%) | 40.55 % | 34.97 % | -5.58 pp | 20.71 % | -19.84 pp | 42.62 % | +2.07 pp |
| DRAM Throughput (%) | 20.60 % | 6.75 % | -13.85 pp | 4.04 % | -16.56 pp | 21.02 % | +0.42 pp |
| Achieved Occupancy (%) | 54.99 % | 73.66 % | +18.67 pp | 68.67 % | +13.68 pp | 55.78 % | +0.79 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.93 % | 54.67 % | -16.26 pp | 35.93 % | -35.00 pp | 71.32 % | +0.39 pp |
| IPC Active | 1.13 inst/cycle | 1.76 inst/cycle | +0.63 | 2.44 inst/cycle | +1.31 | 1.10 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan A: `flat_output_order_iteration_for_coalesced_stores` (1.229×)

Restructuring an NCHW→NHWC permutation kernel to iterate over the flat output (NHWC) index makes stores stride-1 (coalesced) but makes loads strided. In this case the stores became coalesced but the loads became strided, causing L1/TEX hit rate to jump from 0% to 64.5% (loads now served from L1 cache) while excessive sectors on loads increased (62% vs 50% before on stores). The net effect was a modest 1.23x end-to-end speedup despite elapsed cycles increasing (6425→7232), suggesting the L1 cache absorbed the strided load penalty but the overall kernel became more latency-bound (warp cycles per issued instruction rose from 17.4 to 26.1 due to L1TEX scoreboard stalls). This approach is less effective than expected when the kernel is already latency-bound at low DRAM utilization (<7%), because coalescing stores does not address the binding bottleneck.

> **Applicable when**: NCHW→NHWC layout permutation kernel where stores are the dominant uncoalesced access (excessive store sectors >> excessive load sectors), the kernel is bandwidth-bound (DRAM throughput > 20%), and the input tensor fits in L2/L1 cache so strided loads can be absorbed. L1/TEX hit rate near 0% before the change (loads are already coalesced and not cached).

> **Avoid when**: Kernel is latency-bound rather than bandwidth-bound (DRAM throughput < 10%); the input tensor is too large to benefit from L1/L2 caching of strided loads; warp cycles per issued instruction are already high (>20) due to L1TEX stalls, as the strided loads will worsen this further. Also avoid when prior axis-swap attempts have failed correctness — verify index math carefully against actual tensor strides before applying.

---

## Iteration 5

**Best so far**: 1.4938× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_tail_wave_via_l2_grouped_grid_with_correct_autotune_launch | 1.332× | 1.206× | ✓ |  |
| C | fix_hardcoded_grid_to_lambda_to_unlock_autotune | 1.300× | 1.239× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tail_wa... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4990 ns | -1030 ns (-17.1%) | 4860 ns | -1160 ns (-19.3%) |
| SM Throughput (%) | 16.90 % | 25.37 % | +8.47 pp | 25.46 % | +8.56 pp |
| Memory Throughput (%) | 40.55 % | 34.12 % | -6.43 pp | 34.95 % | -5.60 pp |
| DRAM Throughput (%) | 20.60 % | 6.63 % | -13.97 pp | 6.77 % | -13.83 pp |
| Achieved Occupancy (%) | 54.99 % | 73.81 % | +18.82 pp | 74.08 % | +19.09 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 20.00 register/thread | -6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.93 % | 54.67 % | -16.26 pp | 53.75 % | -17.18 pp |
| IPC Active | 1.13 inst/cycle | 1.76 inst/cycle | +0.63 | 1.75 inst/cycle | +0.62 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     4030 ns  (1.494×)  — vectorized_loads_with_autotune

Final best: 5890 ns  (1.0221×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 32}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 32}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +39,15 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    x_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + x_arange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +55,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
