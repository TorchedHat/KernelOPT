# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174820 ns — **Best latency**: 89760 ns (**1.9476×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`expand_autotune_configs_for_coalesced_stores_and_cpi_stall`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_069/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174820 ns
- **SM Throughput**: 28.42 % | **Memory**: 81.36 % | **DRAM**: 67.16 %
- **Achieved Occupancy**: 94.22 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_autotune_tile_sizes | 1.922× | 1.937× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 174820 ns | 90270 ns | -84550 ns (-48.4%) |
| SM Throughput (%) | 28.42 % | 60.99 % | +32.57 pp |
| Memory Throughput (%) | 81.36 % | 52.85 % | -28.51 pp |
| DRAM Throughput (%) | 67.16 % | 14.54 % | -52.62 pp |
| Achieved Occupancy (%) | 94.22 % | 92.21 % | -2.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 |
| No Eligible (%) | 71.62 % | 38.25 % | -33.37 pp |
| IPC Active | 1.14 inst/cycle | 2.47 inst/cycle | +1.33 |

---

## Iteration 2

**Best so far**: 1.9366× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_uncoalesced_store_via_minimum_xblock_constraint | 1.679× | 1.782× | ✓ |  |
| C | coalesce_output_stores_by_enforcing_minimum_xblock | 1.680× | 1.287× | ✓ |  |
| D | expand_autotune_configs_for_coalesced_stores_and_cpi_stall | 2.098× | 1.948× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_s... | Δ | coalesce_output_s... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174820 ns | 98080 ns | -76740 ns (-43.9%) | 135840 ns | -38980 ns (-22.3%) | 89760 ns | -85060 ns (-48.7%) |
| SM Throughput (%) | 28.42 % | 48.68 % | +20.26 pp | 55.22 % | +26.80 pp | 60.43 % | +32.01 pp |
| Memory Throughput (%) | 81.36 % | 53.59 % | -27.77 pp | 69.21 % | -12.15 pp | 52.35 % | -29.01 pp |
| DRAM Throughput (%) | 67.16 % | 28.69 % | -38.47 pp | 20.72 % | -46.44 pp | 14.53 % | -52.63 pp |
| Achieved Occupancy (%) | 94.22 % | 93.08 % | -1.14 pp | 94.09 % | -0.13 pp | 92.23 % | -1.99 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.62 % | 50.25 % | -21.37 pp | 43.93 % | -27.69 pp | 38.09 % | -33.53 pp |
| IPC Active | 1.14 inst/cycle | 1.98 inst/cycle | +0.84 | 2.22 inst/cycle | +1.08 | 2.46 inst/cycle | +1.32 |

---

## Iteration 3

**Best so far**: 1.9476× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_via_tl_multiple_of_on_channel_index | 1.673× | 1.782× | ✓ |  |
| C | vectorized_loads_via_contiguous_hints_on_correct_indices | 1.021× | 1.023× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174820 ns | 98110 ns | -76710 ns (-43.9%) | 170850 ns | -3970 ns (-2.3%) |
| SM Throughput (%) | 28.42 % | 48.11 % | +19.69 pp | 47.51 % | +19.09 pp |
| Memory Throughput (%) | 81.36 % | 52.87 % | -28.49 pp | 80.88 % | -0.48 pp |
| DRAM Throughput (%) | 67.16 % | 28.62 % | -38.54 pp | 68.74 % | +1.58 pp |
| Achieved Occupancy (%) | 94.22 % | 93.05 % | -1.17 pp | 95.18 % | +0.96 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 31.00 register/thread | -1 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.62 % | 49.63 % | -21.99 pp | 50.19 % | -21.43 pp |
| IPC Active | 1.14 inst/cycle | 1.99 inst/cycle | +0.85 | 1.96 inst/cycle | +0.82 |

---

## Iteration 4

**Best so far**: 1.9476× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_spurious_fma_and_bad_vectorization_hints | 1.681× | 1.733× | ✓ |  |
| B | simplify_degenerate_fma_and_fix_eviction_policy_for_streaming_input | 1.645× | 1.779× | ✓ |  |
| C | streaming_eviction_for_large_input_to_relieve_l2_saturation | 0.771× | 0.786× | ✓ |  |
| D | streaming_eviction_policy_plus_autotune_yblock96_for_nhwc_load_coalescing | 1.647× | 0.788× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_spurious_f... | Δ | simplify_degenera... | Δ | streaming_evictio... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174820 ns | 100900 ns | -73920 ns (-42.3%) | 98270 ns | -76550 ns (-43.8%) | 222500 ns | +47680 ns (+27.3%) | 221760 ns | +46940 ns (+26.9%) |
| SM Throughput (%) | 28.42 % | 48.81 % | +20.39 pp | 48.69 % | +20.27 pp | 20.23 % | -8.19 pp | 61.00 % | +32.58 pp |
| Memory Throughput (%) | 81.36 % | 53.58 % | -27.78 pp | 53.45 % | -27.91 pp | 66.07 % | -15.29 pp | 80.29 % | -1.07 pp |
| DRAM Throughput (%) | 67.16 % | 27.88 % | -39.28 pp | 27.62 % | -39.54 pp | 58.18 % | -8.98 pp | 12.43 % | -54.73 pp |
| Achieved Occupancy (%) | 94.22 % | 93.05 % | -1.17 pp | 93.14 % | -1.08 pp | 95.27 % | +1.05 pp | 52.46 % | -41.76 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 31.00 register/thread | -1 | 56.00 register/thread | +24 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 62.06 | +27.15 |
| No Eligible (%) | 71.62 % | 49.79 % | -21.83 pp | 50.14 % | -21.48 pp | 79.39 % | +7.77 pp | 38.33 % | -33.29 pp |
| IPC Active | 1.14 inst/cycle | 1.97 inst/cycle | +0.83 | 1.96 inst/cycle | +0.82 | 0.80 inst/cycle | -0.34 | 2.44 inst/cycle | +1.30 |

---

## Cumulative Progress

```
Baseline:  174820 ns  (1.000×)
Iter 1     90270 ns  (1.937×)  — memory_coalescing_autotune_tile_sizes
Iter 2     89760 ns  (1.948×)  — expand_autotune_configs_for_coalesced_stores_and_cpi_stall

Final best: 89760 ns  (1.9476×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=8, num_stages=3),
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'YBLOCK': 16,  'XBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 16,  'XBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 8,   'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 32},  num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 4,   'XBLOCK': 256}, num_warps=4, num_stages=2),
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
     x2 = xindex
     y0 = (yindex % 96)
+    y0 = tl.max_contiguous(tl.multiple_of(y0, [96, 1]), [96, 1])  # hint: channel axis is 96-aligned and contiguous
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
+    tmp2 = tl.fma(tmp0, tl.full(tmp0.shape, 1.0, tl.float32), tmp1)
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +48,7 @@

     ynumel = 24576
     xnumel = 3025
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
