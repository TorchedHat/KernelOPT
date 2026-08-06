# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_41` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4220 ns — **Best latency**: 4540 ns (**0.9295×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_41`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_41_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 24
- **Duration**: 4220 ns
- **SM Throughput**: 8.70 % | **Memory**: 25.94 % | **DRAM**: 20.38 %
- **Achieved Occupancy**: 38.97 %
- **IPC Active**: 0.73 inst/cycle | **No Eligible**: 80.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | — | 0.929× | ✓ | **✓ best** |
| C | memory_coalescing_swap_tile_dimensions | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4220 ns | 4540 ns | +320 ns (+7.6%) | 4220 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 8.70 % | 9.15 % | +0.45 pp | 8.84 % | +0.14 pp |
| Memory Throughput (%) | 25.94 % | 24.42 % | -1.52 pp | 26.01 % | +0.07 pp |
| DRAM Throughput (%) | 20.38 % | 19.03 % | -1.35 pp | 20.34 % | -0.04 pp |
| Achieved Occupancy (%) | 38.97 % | 39.30 % | +0.33 pp | 39.40 % | +0.43 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 80.58 % | 80.81 % | +0.23 pp | 81.07 % | +0.49 pp |
| IPC Active | 0.73 inst/cycle | 0.74 inst/cycle | +0.01 | 0.72 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_warp_count_and_fix_eviction_policy_for_latency_hiding | — | 1.000× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_latency | — | 0.991× | ✓ |  |
| D | safe_autotune_with_fixed_store_indexing | — | 1.306× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_warp_cou... | Δ | increase_num_warp... | Δ | safe_autotune_wit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4220 ns | 4220 ns | +0 ns (+0.0%) | 4260 ns | +40 ns (+0.9%) | 3230 ns | -990 ns (-23.5%) |
| SM Throughput (%) | 8.70 % | 9.40 % | +0.70 pp | 9.90 % | +1.20 pp | 8.35 % | -0.35 pp |
| Memory Throughput (%) | 25.94 % | 26.18 % | +0.24 pp | 25.75 % | -0.19 pp | 17.94 % | -8.00 pp |
| DRAM Throughput (%) | 20.38 % | 20.45 % | +0.07 pp | 20.20 % | -0.18 pp | 13.31 % | -7.07 pp |
| Achieved Occupancy (%) | 38.97 % | 39.21 % | +0.24 pp | 38.30 % | -0.67 pp | 38.16 % | -0.81 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 16.00 register/thread | -8 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 80.58 % | 81.02 % | +0.44 pp | 80.59 % | +0.01 pp | 80.62 % | +0.04 pp |
| IPC Active | 0.73 inst/cycle | 0.72 inst/cycle | -0.01 | 0.71 inst/cycle | -0.02 | 0.71 inst/cycle | -0.02 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_with_corrected_axis_layout_and_larger_tiles | — | 1.279× | ✓ |  |
| C | remove_fragile_index_hints_and_autotune_tile_sizes | — | 1.319× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_with_cor... | Δ | remove_fragile_in... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4220 ns | 3300 ns | -920 ns (-21.8%) | 3200 ns | -1020 ns (-24.2%) |
| SM Throughput (%) | 8.70 % | 7.55 % | -1.15 pp | 8.94 % | +0.24 pp |
| Memory Throughput (%) | 25.94 % | 17.71 % | -8.23 pp | 17.98 % | -7.96 pp |
| DRAM Throughput (%) | 20.38 % | 13.07 % | -7.31 pp | 13.41 % | -6.97 pp |
| Achieved Occupancy (%) | 38.97 % | 38.35 % | -0.62 pp | 38.07 % | -0.90 pp |
| Registers/Thread | 24.00 register/thread | 16.00 register/thread | -8 | 16.00 register/thread | -8 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 80.58 % | 79.70 % | -0.88 pp | 80.38 % | -0.20 pp |
| IPC Active | 0.73 inst/cycle | 0.72 inst/cycle | -0.01 | 0.78 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  4220 ns  (1.000×)

Final best: 4540 ns  (0.9295×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK':  4}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK':  8}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK':  4}, num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK':  8}, num_warps=16),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 16}, num_warps=16),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=16),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 32
@@ -14,15 +31,13 @@

     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
+    xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 4)
     y1 = yindex // 4
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 131072*y1), ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 32768*y3), tmp0, ymask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 131072*y1), xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x2 + 32768*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

     ynumel = 32
     xnumel = 32768
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_41[grid](arg_0, arg_1, 32, 32768, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_41[grid](arg_0, arg_1, 32, 32768)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
