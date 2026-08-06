# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174620 ns — **Best latency**: 98660 ns (**1.7699×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalesced_nhwc_to_nchw_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_056/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174620 ns
- **SM Throughput**: 27.94 % | **Memory**: 81.52 % | **DRAM**: 67.25 %
- **Achieved Occupancy**: 94.22 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalesced_nhwc_to_nchw_transpose | 1.816× | 1.770× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174620 ns | 98660 ns | -75960 ns (-43.5%) |
| SM Throughput (%) | 27.94 % | 48.67 % | +20.73 pp |
| Memory Throughput (%) | 81.52 % | 53.56 % | -27.96 pp |
| DRAM Throughput (%) | 67.25 % | 28.50 % | -38.75 pp |
| Achieved Occupancy (%) | 94.22 % | 93.07 % | -1.15 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 71.72 % | 52.08 % | -19.64 pp |
| IPC Active | 1.16 inst/cycle | 1.99 inst/cycle | +0.83 |

---

## Iteration 2

**Best so far**: 1.7699× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_tl_max_contiguous_multiple_of_hints | 1.319× | 1.165× | ✓ |  |
| B | expand_autotune_configs_with_deeper_pipelining_and_store_eviction | 1.804× | 1.766× | ✓ |  |
| C | increase_num_stages_to_hide_load_latency_and_fix_store_coalescing | 1.816× | 1.772× | ✓ |  |
| D | eviction_policy_streaming_for_transpose_loads_and_stores | 1.786× | 1.722× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | expand_autotune_c... | Δ | increase_num_stag... | Δ | eviction_policy_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174620 ns | 149890 ns | -24730 ns (-14.2%) | 98880 ns | -75740 ns (-43.4%) | 98530 ns | -76090 ns (-43.6%) | 101410 ns | -73210 ns (-41.9%) |
| SM Throughput (%) | 27.94 % | 39.89 % | +11.95 pp | 49.93 % | +21.99 pp | 48.50 % | +20.56 pp | 48.89 % | +20.95 pp |
| Memory Throughput (%) | 81.52 % | 49.69 % | -31.83 pp | 53.08 % | -28.44 pp | 53.27 % | -28.25 pp | 52.03 % | -29.49 pp |
| DRAM Throughput (%) | 67.25 % | 19.46 % | -47.79 pp | 27.94 % | -39.31 pp | 28.46 % | -38.79 pp | 27.14 % | -40.11 pp |
| Achieved Occupancy (%) | 94.22 % | 94.15 % | -0.07 pp | 92.98 % | -1.24 pp | 92.92 % | -1.30 pp | 92.99 % | -1.23 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.72 % | 60.08 % | -11.64 pp | 49.30 % | -22.42 pp | 50.19 % | -21.53 pp | 48.68 % | -23.04 pp |
| IPC Active | 1.16 inst/cycle | 1.60 inst/cycle | +0.44 | 2.02 inst/cycle | +0.86 | 2.02 inst/cycle | +0.86 | 2.04 inst/cycle | +0.88 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_tl_max_contiguous_multiple_of_hints` (1.165×)

Adding tl.max_contiguous and tl.multiple_of compiler hints to store and load index expressions in Triton kernels can improve L1/TEX cache hit rate (10% → 53%) and L2 hit rate (56% → 76%), reducing DRAM throughput (28.5% → 19.5%) and yielding a 1.165x wall-time speedup despite increased instruction count. However, the hints did not improve coalescing quality for stores — bytes-per-sector for stores actually degraded from 26.3 to 7.1 and excessive sectors grew from 516K to 8.3M, indicating the hints were misapplied to a non-contiguous store pattern (stride-3025 output). The net speedup came from improved cache reuse on loads, not from vectorized stores as intended.

> **Applicable when**: L1/TEX hit rate is very low (<15%) and L2 hit rate is moderate (50-60%), suggesting repeated cache misses on small reused tensors (e.g., bias vectors). The kernel has a small repeatedly-accessed tensor (bias) whose index is a simple modulo expression that the compiler cannot automatically infer as contiguous. Adding tl.max_contiguous + tl.multiple_of on the bias load index can help the compiler reuse cached values.

> **Avoid when**: The store access pattern is non-contiguous (e.g., stride-N output where N >> 1, such as NCHW layout with large spatial stride). Applying tl.max_contiguous to a strided store index (like x2 + 3025*y3 where 3025 is the stride) is semantically incorrect and worsens coalescing — bytes-per-sector dropped from 26.3 to 7.1 and excessive sectors grew 16x. Only apply these hints when the underlying memory access is genuinely contiguous across threads.

---

## Iteration 3

**Best so far**: 1.7699× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_transpose_tile_for_coalesced_io | 1.685× | 1.646× | ✓ |  |
| B | shared_memory_tiled_transpose_for_coalesced_nhwc_to_nchw | 1.630× | 1.768× | ✓ |  |
| C | shared_memory_staged_transpose_to_fix_strided_loads | 1.778× | 1.751× | ✓ |  |
| D | shared_memory_staging_tile_for_transpose_coalescing | 1.804× | 1.761× | ✓ |  |
| E | add_triton_autotune | 1.803× | 1.758× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ | shared_memory_til... | Δ | shared_memory_sta... | Δ | shared_memory_sta... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174620 ns | 106110 ns | -68510 ns (-39.2%) | 98780 ns | -75840 ns (-43.4%) | 99710 ns | -74910 ns (-42.9%) | 99140 ns | -75480 ns (-43.2%) | 99330 ns | -75290 ns (-43.1%) |
| SM Throughput (%) | 27.94 % | 48.92 % | +20.98 pp | 48.90 % | +20.96 pp | 49.98 % | +22.04 pp | 50.02 % | +22.08 pp | 48.38 % | +20.44 pp |
| Memory Throughput (%) | 81.52 % | 46.12 % | -35.40 pp | 49.19 % | -32.33 pp | 53.60 % | -27.92 pp | 53.30 % | -28.22 pp | 53.25 % | -28.27 pp |
| DRAM Throughput (%) | 67.25 % | 26.47 % | -40.78 pp | 27.97 % | -39.28 pp | 27.73 % | -39.52 pp | 27.84 % | -39.41 pp | 28.16 % | -39.09 pp |
| Achieved Occupancy (%) | 94.22 % | 92.87 % | -1.35 pp | 92.75 % | -1.47 pp | 92.99 % | -1.23 pp | 93.10 % | -1.12 pp | 92.93 % | -1.29 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 28.00 register/thread | -4 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.72 % | 50.30 % | -21.42 pp | 50.77 % | -20.95 pp | 48.18 % | -23.54 pp | 48.44 % | -23.28 pp | 49.98 % | -21.74 pp |
| IPC Active | 1.16 inst/cycle | 1.98 inst/cycle | +0.82 | 1.98 inst/cycle | +0.82 | 2.02 inst/cycle | +0.86 | 2.03 inst/cycle | +0.87 | 2.01 inst/cycle | +0.85 |

---

## Cumulative Progress

```
Baseline:  174620 ns  (1.000×)
Iter 1     98660 ns  (1.770×)  — autotune_tile_sizes_for_coalesced_nhwc_to_nchw_transpose

Final best: 98660 ns  (1.7699×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Retained from prior iteration (minus XBLOCK=128,YBLOCK=8,stages=2)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        # NEW: deeper pipeline (num_stages=5) to hide stride-96 load latency
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=5),
+        # NEW: deeper pipeline (num_stages=6)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=6),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=6),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=6),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=6),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=6),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 3025
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,12 +49,10 @@

     y0 = (yindex % 96)
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +61,7 @@

     ynumel = 24576
     xnumel = 3025
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
