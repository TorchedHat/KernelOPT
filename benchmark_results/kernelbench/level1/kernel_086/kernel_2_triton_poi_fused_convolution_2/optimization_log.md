# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174500 ns — **Best latency**: 71780 ns (**2.4310×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`contiguity_hints_and_small_xblock_configs_for_layout_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_086/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174500 ns
- **SM Throughput**: 28.14 % | **Memory**: 81.52 % | **DRAM**: 67.30 %
- **Achieved Occupancy**: 94.16 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 3.123× | 3.225× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalesced_access | 2.842× | 2.432× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174500 ns | 54110 ns | -120390 ns (-69.0%) | 71740 ns | -102760 ns (-58.9%) |
| SM Throughput (%) | 28.14 % | 47.73 % | +19.59 pp | 22.38 % | -5.76 pp |
| Memory Throughput (%) | 81.52 % | 42.86 % | -38.66 pp | 88.03 % | +6.51 pp |
| DRAM Throughput (%) | 67.30 % | 1.38 % | -65.92 pp | 17.01 % | -50.29 pp |
| Achieved Occupancy (%) | 94.16 % | 32.76 % | -61.40 pp | 82.56 % | -11.60 pp |
| Registers/Thread | 32.00 register/thread | 21.00 register/thread | -11 | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.40 % | 49.42 % | -21.98 pp | 76.47 % | +5.07 pp |
| IPC Active | 1.14 inst/cycle | 2.02 inst/cycle | +0.88 | 0.94 inst/cycle | -0.20 |

---

## Iteration 2

**Best so far**: 3.2249× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | prune_uncoalesced_configs_and_expand_num_warps | 2.112× | 1.776× | ✓ |  |
| B | expand_autotune_warps_stages_for_latency_hiding | 3.130× | 3.219× | ✓ |  |
| C | expand_autotune_configs_with_contiguity_hints_for_nhwc_to_nchw_transpose | 3.122× | 3.197× | ✓ |  |
| D | contiguity_hints_and_small_xblock_configs_for_layout_transpose | 3.380× | 2.431× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | prune_uncoalesced... | Δ | expand_autotune_w... | Δ | expand_autotune_c... | Δ | contiguity_hints_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174500 ns | 98270 ns | -76230 ns (-43.7%) | 54210 ns | -120290 ns (-68.9%) | 54590 ns | -119910 ns (-68.7%) | 71780 ns | -102720 ns (-58.9%) |
| SM Throughput (%) | 28.14 % | 48.70 % | +20.56 pp | 47.34 % | +19.20 pp | 47.57 % | +19.43 pp | 22.39 % | -5.75 pp |
| Memory Throughput (%) | 81.52 % | 53.48 % | -28.04 pp | 42.49 % | -39.03 pp | 42.71 % | -38.81 pp | 88.28 % | +6.76 pp |
| DRAM Throughput (%) | 67.30 % | 28.59 % | -38.71 pp | 1.39 % | -65.91 pp | 1.33 % | -65.97 pp | 17.01 % | -50.29 pp |
| Achieved Occupancy (%) | 94.16 % | 92.98 % | -1.18 pp | 31.91 % | -62.25 pp | 31.92 % | -62.24 pp | 82.33 % | -11.83 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 21.00 register/thread | -11 | 21.00 register/thread | -11 | 16.00 register/thread | -16 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.40 % | 49.74 % | -21.66 pp | 50.01 % | -21.39 pp | 49.48 % | -21.92 pp | 76.45 % | +5.05 pp |
| IPC Active | 1.14 inst/cycle | 1.99 inst/cycle | +0.85 | 2.02 inst/cycle | +0.88 | 2.03 inst/cycle | +0.89 | 0.94 inst/cycle | -0.20 |

---

## Iteration 3

**Best so far**: 3.2249× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | channel_aligned_yblock_with_evict_first_load_policy | 3.365× | 1.750× | ✓ |  |
| B | eviction_policy_streaming_and_store_contiguity_hints | 3.355× | 2.486× | ✓ |  |
| D | increase_pipeline_depth_and_evict_first_for_strided_load_latency | 1.972× | 1.770× | ✓ |  |


### Metric Comparison

| Metric | Baseline | channel_aligned_y... | Δ | eviction_policy_s... | Δ | increase_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174500 ns | 99740 ns | -74760 ns (-42.8%) | 70180 ns | -104320 ns (-59.8%) | 98590 ns | -75910 ns (-43.5%) |
| SM Throughput (%) | 28.14 % | 48.47 % | +20.33 pp | 25.66 % | -2.48 pp | 48.14 % | +20.00 pp |
| Memory Throughput (%) | 81.52 % | 53.18 % | -28.34 pp | 92.87 % | +11.35 pp | 52.84 % | -28.68 pp |
| DRAM Throughput (%) | 67.30 % | 27.32 % | -39.98 pp | 16.21 % | -51.09 pp | 27.40 % | -39.90 pp |
| Achieved Occupancy (%) | 94.16 % | 93.04 % | -1.12 pp | 82.25 % | -11.91 pp | 92.97 % | -1.19 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 16.00 register/thread | -16 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.40 % | 50.49 % | -20.91 pp | 73.50 % | +2.10 pp | 49.76 % | -21.64 pp |
| IPC Active | 1.14 inst/cycle | 1.98 inst/cycle | +0.84 | 1.05 inst/cycle | -0.09 | 1.98 inst/cycle | +0.84 |


### Key Learnings

#### ✅ Plan B: `eviction_policy_streaming_and_store_contiguity_hints` (2.486×)

Adding eviction_policy='evict_first' to both the streaming input load and the output store, combined with replacing the autotune config set with small-XBLOCK configs (XBLOCK=1,2,4), caused a regression in this kernel: duration increased from 54.21us to 70.18us (0.77x), L2 throughput exploded from 14% to 93%, and achieved occupancy jumped from 32% to 82% while eligible warps per scheduler dropped from 0.87 to 0.37. The evict_first policy on the output store prevented L2 write-combining, forcing every store to go directly to DRAM and saturating L2 bandwidth. The small-XBLOCK configs (XBLOCK=1,2,4) reintroduced uncoalesced loads (MemoryCacheAccessPattern: 4/32 bytes per sector for global loads from L2, 95% L1 miss rate), undoing the coalescing gains from the prior iteration. The net effect was a slower kernel despite higher occupancy, because the bottleneck shifted from compute/scheduler stalls to L2 bandwidth saturation.

> **Applicable when**: Never: this combination of evict_first on both load and store plus small-XBLOCK autotune configs caused regression. The individual components may be separable but together they are harmful.

> **Avoid when**: When the kernel already achieves low L2 throughput (<20%) and moderate compute throughput (>40%) with XBLOCK>=32 configs: adding evict_first to the output store saturates L2 bandwidth (14%->93%) and adding small-XBLOCK configs (XBLOCK<32) reintroduces uncoalesced loads. Do not apply evict_first to output stores in kernels where L2 write-combining is the primary mechanism keeping L2 throughput low. Do not add XBLOCK<32 configs to autotune when the output store is stride-1 in the x-dimension.

---

## Iteration 4

**Best so far**: 3.2249× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | yblock_contiguity_hints_for_strided_store_and_bias_load | 3.356× | 2.506× | ✓ |  |
| B | expand_autotune_with_small_xblock_large_yblock_configs_for_strided_load_coalescing | 3.357× | 3.202× | ✓ |  |
| C | contiguity_hints_on_yblock_channel_axis_for_vectorized_load | 3.356× | 1.767× | ✓ |  |


### Metric Comparison

| Metric | Baseline | yblock_contiguity... | Δ | expand_autotune_w... | Δ | contiguity_hints_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174500 ns | 69630 ns | -104870 ns (-60.1%) | 54500 ns | -120000 ns (-68.8%) | 98750 ns | -75750 ns (-43.4%) |
| SM Throughput (%) | 28.14 % | 25.68 % | -2.46 pp | 51.05 % | +22.91 pp | 48.17 % | +20.03 pp |
| Memory Throughput (%) | 81.52 % | 93.47 % | +11.95 pp | 42.72 % | -38.80 pp | 52.82 % | -28.70 pp |
| DRAM Throughput (%) | 67.30 % | 16.28 % | -51.02 pp | 1.35 % | -65.95 pp | 27.50 % | -39.80 pp |
| Achieved Occupancy (%) | 94.16 % | 82.62 % | -11.54 pp | 33.41 % | -60.75 pp | 93.04 % | -1.12 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 21.00 register/thread | -11 | 30.00 register/thread | -2 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.40 % | 73.82 % | +2.42 pp | 45.76 % | -25.64 pp | 50.59 % | -20.81 pp |
| IPC Active | 1.14 inst/cycle | 1.04 inst/cycle | -0.10 | 2.15 inst/cycle | +1.01 | 1.96 inst/cycle | +0.82 |

---

## Cumulative Progress

```
Baseline:  174500 ns  (1.000×)
Iter 1     54110 ns  (3.225×)  — vectorized_loads_with_contiguous_hints

Final best: 71780 ns  (2.4310×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,55 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # --- Existing configs (unchanged) ---
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  64, 'YBLOCK':  4}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK':  8}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK':  32, 'YBLOCK':  8}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK':  32, 'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=2),
+        # --- NEW small-XBLOCK configs: channel dim becomes fast-varying -> coalesced loads ---
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 256}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK':  2, 'YBLOCK':  64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  2, 'YBLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  32}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  64}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 128}, num_warps=8, num_stages=4),
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
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(
+        tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK
+    )
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 96)
     y1 = yindex // 96
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +62,7 @@

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
