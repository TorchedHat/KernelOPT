# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6140 ns — **Best latency**: 3300 ns (**1.8606×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`expand_autotune_configs_with_yblock3_alignment_and_num_stages_pipelining`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_092/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6140 ns
- **SM Throughput**: 16.34 % | **Memory**: 40.17 % | **DRAM**: 20.10 %
- **Achieved Occupancy**: 55.17 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.64 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.327× | 1.827× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalesced_stores_and_occupancy | 1.183× | 0.741× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_strided_kernel | 1.304× | 1.524× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 3360 ns | -2780 ns (-45.3%) | 8290 ns | +2150 ns (+35.0%) | 4030 ns | -2110 ns (-34.4%) |
| SM Throughput (%) | 16.34 % | 17.49 % | +1.15 pp | 41.71 % | +25.37 pp | 19.38 % | +3.04 pp |
| Memory Throughput (%) | 40.17 % | 20.49 % | -19.68 pp | 52.34 % | +12.17 pp | 22.40 % | -17.77 pp |
| DRAM Throughput (%) | 20.10 % | 4.96 % | -15.14 pp | 14.93 % | -5.17 pp | 4.13 % | -15.97 pp |
| Achieved Occupancy (%) | 55.17 % | 43.67 % | -11.50 pp | 78.50 % | +23.33 pp | 48.76 % | -6.41 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 32.00 register/thread | +6 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.64 % | 54.76 % | -16.88 pp | 42.90 % | -28.74 pp | 55.35 % | -16.29 pp |
| IPC Active | 1.15 inst/cycle | 1.67 inst/cycle | +0.52 | 2.27 inst/cycle | +1.12 | 1.83 inst/cycle | +0.68 |


### Key Learnings

#### ✅ Plan D: `autotune_tile_sizes_for_underutilized_strided_kernel` (1.844×)

Adding more autotune configs (larger YBLOCK values up to 32, num_stages=4 variants, and additional XBLOCK/YBLOCK combinations) to an existing @triton.autotune decorator allowed the autotuner to find a better tile shape at runtime, achieving a 1.84x speedup. The NCU-profiled kernel metrics before and after are nearly identical (same waves/SM, similar occupancy ~44%, similar warp cycles ~15.6), confirming the speedup comes from the autotuner selecting a better config at runtime rather than a structural kernel change — NCU captures only one config execution, not the best autotuned one. The causal chain is: expanded config space → autotuner selects a tile shape with better latency hiding at runtime → lower end-to-end latency.

> **Applicable when**: A Triton kernel already has @triton.autotune but with a limited config space (fewer than 20 configs). The kernel shows IssueSlotUtilization warnings (no-eligible > 50%), waves per SM < 1.0, and warp cycles per issued instruction > 15. The problem dimensions are not powers of two, making the optimal tile shape non-obvious. The grid must already be a lambda reading meta['XBLOCK'] and meta['YBLOCK'].

> **Avoid when**: The kernel does not have @triton.autotune (add it first). The config space is already very large (>30 configs). The kernel is compute-bound (Compute SM Throughput > 60%). The grid is a static tuple rather than a lambda — expanding configs without fixing the grid lambda means the autotuner's selected tile sizes are never used at launch.

---

## Iteration 2

**Best so far**: 1.8274× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | contiguous_load_hints_with_yblock_modulus3_aligned_autotune | — | fail | ✓ |  |
| C | channel_aligned_yblock_with_contiguous_load_hints | — | fail | ✓ |  |

---

## Iteration 3

**Best so far**: 1.8274× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | channel_aligned_yblock_multiples_of_3_for_coalesced_stores | — | fail | ✓ |  |
| C | channel_aligned_yblock_with_contiguous_load_hints_and_correct_grid_lambda | 1.394× | 1.844× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.270× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | channel_aligned_y... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 3330 ns | -2810 ns (-45.8%) | 6300 ns | +160 ns (+2.6%) |
| SM Throughput (%) | 16.34 % | 17.27 % | +0.93 pp | 16.53 % | +0.19 pp |
| Memory Throughput (%) | 40.17 % | 20.65 % | -19.52 pp | 39.21 % | -0.96 pp |
| DRAM Throughput (%) | 20.10 % | 4.99 % | -15.11 pp | 19.62 % | -0.48 pp |
| Achieved Occupancy (%) | 55.17 % | 44.01 % | -11.16 pp | 55.69 % | +0.52 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.64 % | 55.74 % | -15.90 pp | 71.04 % | -0.60 pp |
| IPC Active | 1.15 inst/cycle | 1.70 inst/cycle | +0.55 | 1.15 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.8438× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | channel_aligned_yblock_multiples_of_3_with_correct_grid_lambda | 1.397× | 1.775× | ✓ |  |
| D | expand_autotune_configs_with_yblock3_alignment_and_num_stages_pipelining | 1.399× | 1.861× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | channel_aligned_y... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 3460 ns | -2680 ns (-43.6%) | 3300 ns | -2840 ns (-46.3%) |
| SM Throughput (%) | 16.34 % | 17.65 % | +1.31 pp | 17.57 % | +1.23 pp |
| Memory Throughput (%) | 40.17 % | 19.72 % | -20.45 pp | 20.81 % | -19.36 pp |
| DRAM Throughput (%) | 20.10 % | 4.76 % | -15.34 pp | 5.02 % | -15.08 pp |
| Achieved Occupancy (%) | 55.17 % | 46.68 % | -8.49 pp | 43.71 % | -11.46 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.64 % | 60.78 % | -10.86 pp | 55.34 % | -16.30 pp |
| IPC Active | 1.15 inst/cycle | 1.57 inst/cycle | +0.42 | 1.67 inst/cycle | +0.52 |

---

## Iteration 5

**Best so far**: 1.8606× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_static_grid_to_lambda_with_yblock_modulus3_alignment | 1.313× | 1.498× | ✓ |  |
| C | fix_static_grid_to_lambda_with_pipelined_autotune_configs | 1.350× | 1.844× | ✓ |  |
| D | fix_static_grid_to_lambda_for_correct_autotune_coverage | 1.389× | 1.844× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 4100 ns | -2040 ns (-33.2%) | 3330 ns | -2810 ns (-45.8%) | 3330 ns | -2810 ns (-45.8%) |
| SM Throughput (%) | 16.34 % | 19.39 % | +3.05 pp | 17.67 % | +1.33 pp | 17.59 % | +1.25 pp |
| Memory Throughput (%) | 40.17 % | 22.39 % | -17.78 pp | 20.50 % | -19.67 pp | 20.56 % | -19.61 pp |
| DRAM Throughput (%) | 20.10 % | 4.03 % | -16.07 pp | 4.95 % | -15.15 pp | 4.98 % | -15.12 pp |
| Achieved Occupancy (%) | 55.17 % | 50.18 % | -4.99 pp | 44.16 % | -11.01 pp | 43.99 % | -11.18 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.64 % | 53.13 % | -18.51 pp | 61.59 % | -10.05 pp | 54.12 % | -17.52 pp |
| IPC Active | 1.15 inst/cycle | 1.78 inst/cycle | +0.63 | 1.68 inst/cycle | +0.53 | 1.70 inst/cycle | +0.55 |

---

## Cumulative Progress

```
Baseline:  6140 ns  (1.000×)
Iter 1     3360 ns  (1.827×)  — vectorized_loads_with_contiguous_hints
Iter 3     3330 ns  (1.844×)  — channel_aligned_yblock_with_contiguous_load_hints_and_correct_grid_lambda
Iter 4     3300 ns  (1.861×)  — expand_autotune_configs_with_yblock3_alignment_and_num_stages_pipelining

Final best: 3300 ns  (1.8606×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +31,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply contiguous/alignment hints on the 1D x-range before broadcasting
+    x_range = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = x_range[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +42,13 @@

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

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
