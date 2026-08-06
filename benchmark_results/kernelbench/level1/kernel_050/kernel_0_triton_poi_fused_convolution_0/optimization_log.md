# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 4260 ns (**1.4272×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_to_use_meta_and_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_050/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 16.82 % | **Memory**: 40.36 % | **DRAM**: 20.30 %
- **Achieved Occupancy**: 54.93 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_tile_size_increase | 1.208× | 0.891× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.305× | 1.000× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_swap | 1.331× | 0.970× | ✓ |  |
| D | autotune_tile_sizes_for_layout_permutation_kernel | 1.381× | 1.520× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 6820 ns | +740 ns (+12.2%) | 6080 ns | +0 ns (+0.0%) | 6270 ns | +190 ns (+3.1%) | 4000 ns | -2080 ns (-34.2%) |
| SM Throughput (%) | 16.82 % | 29.20 % | +12.38 pp | 16.56 % | -0.26 pp | 14.51 % | -2.31 pp | 26.04 % | +9.22 pp |
| Memory Throughput (%) | 40.36 % | 38.18 % | -2.18 pp | 41.16 % | +0.80 pp | 26.40 % | -13.96 pp | 20.64 % | -19.72 pp |
| DRAM Throughput (%) | 20.30 % | 18.03 % | -2.27 pp | 20.30 % | +0.00 pp | 4.94 % | -15.36 pp | 4.13 % | -16.17 pp |
| Achieved Occupancy (%) | 54.93 % | 75.70 % | +20.77 pp | 55.12 % | +0.19 pp | 42.85 % | -12.08 pp | 68.36 % | +13.43 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 26.00 register/thread | +0 | 24.00 register/thread | -2 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.47 % | 54.62 % | -15.85 pp | 70.74 % | +0.27 pp | 68.61 % | -1.86 pp | 38.39 % | -32.08 pp |
| IPC Active | 1.15 inst/cycle | 1.88 inst/cycle | +0.73 | 1.15 inst/cycle | +0.00 | 1.19 inst/cycle | +0.04 | 2.39 inst/cycle | +1.24 |

---

## Iteration 2

**Best so far**: 1.52× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.401× | 1.407× | ✓ |  |
| B | fix_autotune_grid_and_add_vectorized_load_hints | 1.295× | 1.050× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.447× | 1.327× | ✓ |  |
| D | fix_autotune_grid_lambda_and_expand_configs_to_eliminate_partial_wave | 1.472× | 0.905× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4320 ns | -1760 ns (-28.9%) | 5790 ns | -290 ns (-4.8%) | 4580 ns | -1500 ns (-24.7%) | 6720 ns | +640 ns (+10.5%) |
| SM Throughput (%) | 16.82 % | 28.79 % | +11.97 pp | 28.11 % | +11.29 pp | 27.98 % | +11.16 pp | 29.25 % | +12.43 pp |
| Memory Throughput (%) | 40.36 % | 21.29 % | -19.07 pp | 30.79 % | -9.57 pp | 22.19 % | -18.17 pp | 38.55 % | -1.81 pp |
| DRAM Throughput (%) | 20.30 % | 3.84 % | -16.46 pp | 2.85 % | -17.45 pp | 3.60 % | -16.70 pp | 18.37 % | -1.93 pp |
| Achieved Occupancy (%) | 54.93 % | 69.10 % | +14.17 pp | 72.84 % | +17.91 pp | 67.21 % | +12.28 pp | 75.95 % | +21.02 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 18.00 register/thread | -8 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.47 % | 43.57 % | -26.90 pp | 41.35 % | -29.12 pp | 37.42 % | -33.05 pp | 51.60 % | -18.87 pp |
| IPC Active | 1.15 inst/cycle | 2.43 inst/cycle | +1.28 | 2.44 inst/cycle | +1.29 | 2.36 inst/cycle | +1.21 | 1.89 inst/cycle | +0.74 |

---

## Iteration 3

**Best so far**: 1.52× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_and_eliminate_partial_wave | 1.398× | 1.427× | ✓ |  |
| B | eliminate_partial_wave_via_grid_lambda_fix | 1.476× | 1.427× | ✓ | **✓ best** |
| C | fix_hardcoded_grid_and_eviction_policy_for_layout_permutation | 1.264× | 1.097× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | eliminate_partial... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4260 ns | -1820 ns (-29.9%) | 4260 ns | -1820 ns (-29.9%) | 5540 ns | -540 ns (-8.9%) |
| SM Throughput (%) | 16.82 % | 28.45 % | +11.63 pp | 27.48 % | +10.66 pp | 28.26 % | +11.44 pp |
| Memory Throughput (%) | 40.36 % | 21.02 % | -19.34 pp | 21.84 % | -18.52 pp | 30.96 % | -9.40 pp |
| DRAM Throughput (%) | 20.30 % | 3.86 % | -16.44 pp | 3.89 % | -16.41 pp | 2.98 % | -17.32 pp |
| Achieved Occupancy (%) | 54.93 % | 68.75 % | +13.82 pp | 68.76 % | +13.83 pp | 73.31 % | +18.38 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.47 % | 37.34 % | -33.13 pp | 37.52 % | -32.95 pp | 37.40 % | -33.07 pp |
| IPC Active | 1.15 inst/cycle | 2.36 inst/cycle | +1.21 | 2.40 inst/cycle | +1.25 | 2.44 inst/cycle | +1.29 |

---

## Iteration 4

**Best so far**: 1.52× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | large_tile_configs_to_eliminate_partial_wave | 1.475× | 0.069× | ✓ |  |
| C | fix_broken_autotune_grid_with_wave_aligned_configs | 1.403× | 1.067× | ✓ |  |


### Metric Comparison

| Metric | Baseline | large_tile_config... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 87740 ns | +81660 ns (+1343.1%) | 5700 ns | -380 ns (-6.2%) |
| SM Throughput (%) | 16.82 % | 41.70 % | +24.88 pp | 26.35 % | +9.53 pp |
| Memory Throughput (%) | 40.36 % | 71.22 % | +30.86 pp | 28.87 % | -11.49 pp |
| DRAM Throughput (%) | 20.30 % | 0.23 % | -20.07 pp | 2.90 % | -17.40 pp |
| Achieved Occupancy (%) | 54.93 % | 24.66 % | -30.27 pp | 74.30 % | +19.37 pp |
| Registers/Thread | 26.00 register/thread | 128.00 register/thread | +102 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 11.88 | +11.14 | 1.48 | +0.74 |
| No Eligible (%) | 70.47 % | 56.00 % | -14.47 pp | 37.70 % | -32.77 pp |
| IPC Active | 1.15 inst/cycle | 1.75 inst/cycle | +0.60 | 2.35 inst/cycle | +1.20 |

---

## Iteration 5

**Best so far**: 1.52× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_wave_aligned_autotune_configs | 1.049× | 0.838× | ✓ |  |
| B | 1d_flattened_wave_aligned_kernel_rewrite | 1.247× | 0.534× | ✓ |  |
| C | purge_oversized_autotune_configs_to_eliminate_register_spill | 1.457× | 1.520× | ✓ |  |
| D | purge_register_spilling_autotune_configs | 1.473× | 1.483× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | 1d_flattened_wave... | Δ | purge_oversized_a... | Δ | purge_register_sp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 7260 ns | +1180 ns (+19.4%) | 11390 ns | +5310 ns (+87.3%) | 4000 ns | -2080 ns (-34.2%) | 4100 ns | -1980 ns (-32.6%) |
| SM Throughput (%) | 16.82 % | 37.52 % | +20.70 pp | 36.85 % | +20.03 pp | 26.88 % | +10.06 pp | 27.35 % | +10.53 pp |
| Memory Throughput (%) | 40.36 % | 48.42 % | +8.06 pp | 53.93 % | +13.57 pp | 21.37 % | -18.99 pp | 21.72 % | -18.64 pp |
| DRAM Throughput (%) | 20.30 % | 4.55 % | -15.75 pp | 1.45 % | -18.85 pp | 4.12 % | -16.18 pp | 4.05 % | -16.25 pp |
| Achieved Occupancy (%) | 54.93 % | 76.01 % | +21.08 pp | 53.75 % | -1.18 pp | 68.02 % | +13.09 pp | 68.61 % | +13.68 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 46.00 register/thread | +20 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 2.38 | +1.64 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.47 % | 38.23 % | -32.24 pp | 43.15 % | -27.32 pp | 37.54 % | -32.93 pp | 42.61 % | -27.86 pp |
| IPC Active | 1.15 inst/cycle | 2.46 inst/cycle | +1.31 | 2.30 inst/cycle | +1.15 | 2.37 inst/cycle | +1.22 | 2.38 inst/cycle | +1.23 |


### Key Learnings

#### ✅ Plan C: `purge_oversized_autotune_configs_to_eliminate_register_spill` (1.520×)

When a Triton autotune config list includes tile sizes (XBLOCK >= 2048) that push registers_per_thread above the safe threshold (~18-25), the autotuner may select a register-spilling config, inflating instruction count and collapsing memory throughput. Removing configs with XBLOCK >= 2048 (and any other oversized tiles) from the autotune list forces the autotuner to select a register-safe config, reducing registers_per_thread from 25 to 18, cutting executed instructions from 1.24M to 869K (~30%), and improving memory throughput from 141.7 GB/s to 201.6 GB/s. The causal chain is: oversized tile → elevated register pressure → increased instruction count and L1/TEX cache pressure → lower effective memory throughput and higher elapsed cycles.

> **Applicable when**: Triton kernel has @triton.autotune with configs including XBLOCK >= 2048, registers_per_thread is elevated (>20) relative to a plain-arange baseline, memory throughput is below expected for the available bandwidth, and executed instruction count is higher than expected for the problem size. The kernel body is a simple load/store (elementwise or permutation) without heavy compute.

> **Avoid when**: All autotune configs already have XBLOCK <= 1024 and registers_per_thread is already at the minimum (~18). Do not apply if the kernel is compute-bound (Compute SM Throughput > 60%) rather than memory-bound — in that case, larger tiles may improve cache reuse and reduce grid overhead. Do not apply if the kernel has already been profiled and the winning config is a small-tile config.

#### ✅ Plan D: `purge_register_spilling_autotune_configs` (1.483×)

When a Triton autotune config list contains XBLOCK values (e.g., ≥2048) that force the compiler to allocate near the maximum register budget per thread, the selected config can cause register spilling, inflating executed instructions and kernel duration. Removing configs with XBLOCK above a safe threshold (≤1024 for typical 2D tiled pointwise kernels) forces autotune to select register-safe configs, reducing registers_per_thread (25→18), cutting executed instructions (~30%), and improving memory throughput (141→197 GB/s) by reducing L1/TEX pressure from spill traffic. The causal chain is: oversized XBLOCK → high register count → spill to local memory → inflated L1/TEX throughput from spill traffic → increased instruction count and latency.

> **Applicable when**: Triton kernel has @triton.autotune with configs containing XBLOCK >= 2048 (or any value that pushes registers_per_thread above ~20-25 for the given num_warps). Profiling shows elevated registers_per_thread, high L1/TEX Cache Throughput from spill traffic (>60%), and executed instructions significantly above the theoretical minimum for the workload. Block Limit Registers is low (≤8 blocks/SM), indicating register pressure is constraining occupancy.

> **Avoid when**: All autotune configs already have XBLOCK ≤ 1024 and registers_per_thread is already at or below 20. Do not apply if the kernel is compute-bound (Compute SM Throughput > 60%) and large XBLOCK configs are needed for arithmetic intensity. Do not apply if the kernel has been verified to not spill (local load/store bytes = 0 in NCU).

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     4000 ns  (1.520×)  — autotune_tile_sizes_for_layout_permutation_kernel

Final best: 4260 ns  (1.4272×)
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
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +31,15 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    x_arange = tl.arange(0, XBLOCK)
+    xindex = xoffset + x_arange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +47,7 @@

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
