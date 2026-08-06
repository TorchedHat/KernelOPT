# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5150 ns — **Best latency**: 5220 ns (**0.9866×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_32`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_32_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 27
- **Duration**: 5150 ns
- **SM Throughput**: 34.19 % | **Memory**: 35.10 % | **DRAM**: 2.11 %
- **Achieved Occupancy**: 72.13 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 44.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_corrected_tile_sizes | — | 0.987× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5150 ns | 5220 ns | +70 ns (+1.4%) |
| SM Throughput (%) | 34.19 % | 34.25 % | +0.06 pp |
| Memory Throughput (%) | 35.10 % | 35.19 % | +0.09 pp |
| DRAM Throughput (%) | 2.11 % | 2.08 % | -0.03 pp |
| Achieved Occupancy (%) | 72.13 % | 70.96 % | -1.17 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 44.99 % | 39.00 % | -5.99 pp |
| IPC Active | 2.40 inst/cycle | 2.26 inst/cycle | -0.14 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_correct_index_arithmetic | — | 0.975× | ✓ |  |
| C | memory_coalescing_1d_flat_kernel_rewrite | — | 1.134× | ✓ |  |
| D | increase_yblock_to_eliminate_partial_wave_tail | — | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | increase_yblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5150 ns | 5280 ns | +130 ns (+2.5%) | 4540 ns | -610 ns (-11.8%) | 5090 ns | -60 ns (-1.2%) |
| SM Throughput (%) | 34.19 % | 34.04 % | -0.15 pp | 16.20 % | -17.99 pp | 33.22 % | -0.97 pp |
| Memory Throughput (%) | 35.10 % | 34.96 % | -0.14 pp | 6.49 % | -28.61 pp | 34.13 % | -0.97 pp |
| DRAM Throughput (%) | 2.11 % | 2.05 % | -0.06 pp | 2.39 % | +0.28 pp | 2.13 % | +0.02 pp |
| Achieved Occupancy (%) | 72.13 % | 70.53 % | -1.60 pp | 30.86 % | -41.27 pp | 68.83 % | -3.30 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 16.00 register/thread | -11 | 27.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 44.99 % | 43.52 % | -1.47 pp | 53.85 % | +8.86 pp | 43.46 % | -1.53 pp |
| IPC Active | 2.40 inst/cycle | 2.24 inst/cycle | -0.16 | 1.65 inst/cycle | -0.75 | 2.37 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_num_warps_for_latency_hiding | — | 1.176× | ✓ |  |
| B | autotune_2d_kernel_remove_broken_vectorization_hints | — | 1.150× | ✓ |  |
| C | autotune_2d_kernel_remove_broken_vectorization_hints | — | 1.150× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | autotune_2d_kerne... | Δ | autotune_2d_kerne... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5150 ns | 4380 ns | -770 ns (-15.0%) | 4480 ns | -670 ns (-13.0%) | 4480 ns | -670 ns (-13.0%) |
| SM Throughput (%) | 34.19 % | 3.70 % | -30.49 pp | 16.10 % | -18.09 pp | 16.29 % | -17.90 pp |
| Memory Throughput (%) | 35.10 % | 2.46 % | -32.64 pp | 6.44 % | -28.66 pp | 6.51 % | -28.59 pp |
| DRAM Throughput (%) | 2.11 % | 2.46 % | +0.35 pp | 2.40 % | +0.29 pp | 2.41 % | +0.30 pp |
| Achieved Occupancy (%) | 72.13 % | 11.28 % | -60.85 pp | 31.26 % | -40.87 pp | 32.10 % | -40.03 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 | 16.00 register/thread | -11 | 16.00 register/thread | -11 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 44.99 % | 81.49 % | +36.50 pp | 54.07 % | +9.08 pp | 53.56 % | +8.57 pp |
| IPC Active | 2.40 inst/cycle | 0.49 inst/cycle | -1.91 | 1.65 inst/cycle | -0.75 | 1.64 inst/cycle | -0.76 |


### Key Learnings

#### ✅ Plan B: `autotune_2d_kernel_remove_broken_vectorization_hints` (1.150×)

When a Triton kernel has XBLOCK much larger than xnumel (e.g. XBLOCK=128 for xnumel=4), most threads are masked out and the grid creates a large partial wave. Adding @triton.autotune with configs spanning the correct XBLOCK/YBLOCK range allows the autotuner to select tile sizes that match the actual problem dimensions. Critically, tl.max_contiguous/tl.multiple_of hints that are only valid for a specific XBLOCK value must be removed entirely — leaving them in place causes CUDA errors when the autotuner selects other XBLOCK values. The ymask must also be corrected from tl.full([YBLOCK], True, tl.int1) to yindex < ynumel to properly handle bounds checking with larger YBLOCK values.

> **Applicable when**: A 2D Triton kernel has XBLOCK >> xnumel (e.g. XBLOCK=128 for xnumel=4), causing most threads to be masked out and creating a large partial-wave grid. NCU flags LaunchConfiguration (up to 50% tail penalty) and MemoryCacheAccessPattern (low bytes-per-sector utilization). The kernel also uses tl.max_contiguous/tl.multiple_of hints tied to a specific XBLOCK value, and/or ymask=tl.full([YBLOCK],True,tl.int1) that suppresses bounds checking.

> **Avoid when**: The kernel's tl.max_contiguous/tl.multiple_of hints are valid across all autotune configs (i.e. XBLOCK is always the same value). Do not apply if the index arithmetic is sensitive to YBLOCK changes in ways that cannot be handled by proper ymask bounds checking. Note: the actual profiling metrics changed only marginally (Duration 4.54→4.48 us, Occupancy 30.86→31.26%) — the speedup of 1.15x likely reflects end-to-end timing improvements from the autotuner selecting better configs, not large per-metric changes visible in NCU.

#### ✅ Plan C: `autotune_2d_kernel_remove_broken_vectorization_hints` (1.150×)

When a Triton kernel has XBLOCK much larger than xnumel (e.g. XBLOCK=128 for xnumel=4), most threads are masked out and the grid creates a large partial wave. Adding @triton.autotune with configs spanning the correct XBLOCK/YBLOCK range allows the autotuner to select tile sizes that match the actual problem dimensions. Critically, tl.max_contiguous/tl.multiple_of hints that are only valid for a specific XBLOCK value must be removed entirely — leaving them in place causes CUDA errors when the autotuner selects other XBLOCK values. The ymask must also be corrected from tl.full([YBLOCK], True, tl.int1) to yindex < ynumel to properly handle bounds checking with larger YBLOCK values.

> **Applicable when**: A 2D Triton kernel has XBLOCK >> xnumel (e.g. XBLOCK=128 for xnumel=4), causing most threads to be masked out and creating a large partial-wave grid. NCU flags LaunchConfiguration (up to 50% tail penalty) and MemoryCacheAccessPattern (low bytes-per-sector utilization). The kernel also uses tl.max_contiguous/tl.multiple_of hints tied to a specific XBLOCK value, and/or ymask=tl.full([YBLOCK],True,tl.int1) that suppresses bounds checking.

> **Avoid when**: The kernel's tl.max_contiguous/tl.multiple_of hints are valid across all autotune configs (i.e. XBLOCK is always the same value). Do not apply if the index arithmetic is sensitive to YBLOCK changes in ways that cannot be handled by proper ymask bounds checking. Note: the actual profiling metrics changed only marginally (Duration 4.54→4.48 us, Occupancy 30.86→31.26%) — the speedup of 1.15x likely reflects end-to-end timing improvements from the autotuner selecting better configs, not large per-metric changes visible in NCU.

---

## Cumulative Progress

```
Baseline:  5150 ns  (1.000×)

Final best: 5220 ns  (0.9866×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 64, 'XBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 4}, num_warps=8, num_stages=1),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 4}, num_warps=8, num_stages=1),
+        triton.Config({'YBLOCK': 512, 'XBLOCK': 4}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 32768
     xnumel = 4
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), mask=ymask & xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, mask=ymask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 32768
     xnumel = 4
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_32[grid](arg_0, arg_1, 32768, 4, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_32[grid](arg_0, arg_1, 32768, 4)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
