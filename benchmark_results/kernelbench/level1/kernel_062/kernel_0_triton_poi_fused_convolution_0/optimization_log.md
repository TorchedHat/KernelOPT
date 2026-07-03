# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 4130 ns (**1.4479×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`stride3_aligned_yblock_configs_with_vectorized_load_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_062/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.37 % | **Memory**: 41.22 % | **DRAM**: 20.65 %
- **Achieved Occupancy**: 55.11 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.394× | 1.404× | ✓ |  |
| B | fix_uncoalesced_stores_via_autotune_tile_reshape | 1.425× | 1.375× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_swap | 1.262× | 0.613× | ✓ |  |
| D | add_autotune_increase_tile_sizes_and_warps | 1.113× | 0.968× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ | memory_coalescing... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4260 ns | -1720 ns (-28.8%) | 4350 ns | -1630 ns (-27.3%) | 9760 ns | +3780 ns (+63.2%) | 6180 ns | +200 ns (+3.3%) |
| SM Throughput (%) | 16.37 % | 26.81 % | +10.44 pp | 26.71 % | +10.34 pp | 13.40 % | -2.97 pp | 16.56 % | +0.19 pp |
| Memory Throughput (%) | 41.22 % | 21.28 % | -19.94 pp | 20.11 % | -21.11 pp | 31.01 % | -10.21 pp | 40.92 % | -0.30 pp |
| DRAM Throughput (%) | 20.65 % | 3.90 % | -16.75 pp | 3.81 % | -16.84 pp | 6.31 % | -14.34 pp | 20.05 % | -0.60 pp |
| Achieved Occupancy (%) | 55.11 % | 68.21 % | +13.10 pp | 70.13 % | +15.02 pp | 52.62 % | -2.49 pp | 55.39 % | +0.28 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 32.00 register/thread | +6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.71 % | 39.46 % | -31.25 pp | 37.35 % | -33.36 pp | 73.62 % | +2.91 pp | 70.58 % | -0.13 pp |
| IPC Active | 1.16 inst/cycle | 2.39 inst/cycle | +1.23 | 2.47 inst/cycle | +1.31 | 1.02 inst/cycle | -0.14 | 1.12 inst/cycle | -0.04 |

---

## Iteration 2

**Best so far**: 1.3747× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_store_policy_aligned_yblock_and_num_stages | 1.408× | 1.404× | ✓ |  |
| B | align_yblock_to_stride_period_with_load_contiguity_hints | 1.373× | 1.404× | ✓ |  |
| C | autotune_stride_aligned_yblock_with_evict_first | 1.342× | 1.394× | ✓ |  |
| D | autotune_yblock_aligned_to_stride_period | 1.440× | 1.438× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_store... | Δ | align_yblock_to_s... | Δ | autotune_stride_a... | Δ | autotune_yblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4260 ns | -1720 ns (-28.8%) | 4260 ns | -1720 ns (-28.8%) | 4290 ns | -1690 ns (-28.3%) | 4160 ns | -1820 ns (-30.4%) |
| SM Throughput (%) | 16.37 % | 28.05 % | +11.68 pp | 27.58 % | +11.21 pp | 27.46 % | +11.09 pp | 28.16 % | +11.79 pp |
| Memory Throughput (%) | 41.22 % | 21.08 % | -20.14 pp | 20.70 % | -20.52 pp | 20.61 % | -20.61 pp | 21.16 % | -20.06 pp |
| DRAM Throughput (%) | 20.65 % | 3.88 % | -16.77 pp | 3.87 % | -16.78 pp | 3.87 % | -16.78 pp | 3.98 % | -16.67 pp |
| Achieved Occupancy (%) | 55.11 % | 68.21 % | +13.10 pp | 68.07 % | +12.96 pp | 68.74 % | +13.63 pp | 69.77 % | +14.66 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.71 % | 37.13 % | -33.58 pp | 36.43 % | -34.28 pp | 36.84 % | -33.87 pp | 36.62 % | -34.09 pp |
| IPC Active | 1.16 inst/cycle | 2.39 inst/cycle | +1.23 | 2.43 inst/cycle | +1.27 | 2.39 inst/cycle | +1.23 | 2.32 inst/cycle | +1.16 |

---

## Iteration 3

**Best so far**: 1.4375× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_yblock30_full_y_coverage | 1.295× | 0.205× | ✓ |  |
| B | yblock_multiple_of_stride_period_to_coalesce_stores | 1.417× | 1.079× | ✓ |  |
| C | wave_aligned_persistent_1d_grid_to_eliminate_partial_wave | 0.470× | 0.372× | ✓ |  |
| D | fix_autotune_configs_with_stride3_aligned_yblock_and_meta_grid | 1.341× | 1.427× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | yblock_multiple_o... | Δ | wave_aligned_pers... | Δ | fix_autotune_conf... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 29220 ns | +23240 ns (+388.6%) | 5540 ns | -440 ns (-7.4%) | 16060 ns | +10080 ns (+168.6%) | 4190 ns | -1790 ns (-29.9%) |
| SM Throughput (%) | 16.37 % | 60.34 % | +43.97 pp | 27.33 % | +10.96 pp | 32.01 % | +15.64 pp | 26.15 % | +9.78 pp |
| Memory Throughput (%) | 41.22 % | 73.29 % | +32.07 pp | 28.83 % | -12.39 pp | 61.99 % | +20.77 pp | 19.64 % | -21.58 pp |
| DRAM Throughput (%) | 20.65 % | 4.21 % | -16.44 pp | 5.58 % | -15.07 pp | 7.67 % | -12.98 pp | 3.94 % | -16.71 pp |
| Achieved Occupancy (%) | 55.11 % | 34.45 % | -20.66 pp | 82.34 % | +27.23 pp | 51.67 % | -3.44 pp | 69.40 % | +14.29 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 22.00 register/thread | -4 | 48.00 register/thread | +22 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 3.96 | +3.22 | 1.48 | +0.74 | 1.19 | +0.45 | 1.48 | +0.74 |
| No Eligible (%) | 70.71 % | 48.39 % | -22.32 pp | 47.03 % | -23.68 pp | 56.73 % | -13.98 pp | 37.13 % | -33.58 pp |
| IPC Active | 1.16 inst/cycle | 2.03 inst/cycle | +0.87 | 1.94 inst/cycle | +0.78 | 1.74 inst/cycle | +0.58 | 2.44 inst/cycle | +1.28 |

---

## Iteration 4

**Best so far**: 1.4375× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | stride3_aligned_yblock_configs_with_vectorized_load_hints | 1.460× | 1.448× | ✓ | **✓ best** |
| C | revert_persistent_loop_to_2d_grid_with_autotune | 1.248× | 1.417× | ✓ |  |
| D | restore_occupancy_by_removing_smem_exploding_yblock_configs | 1.403× | 1.384× | ✓ |  |
| E | add_triton_autotune | 1.045× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | stride3_aligned_y... | Δ | revert_persistent... | Δ | restore_occupancy... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4130 ns | -1850 ns (-30.9%) | 4220 ns | -1760 ns (-29.4%) | 4320 ns | -1660 ns (-27.8%) | 5980 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 16.37 % | 26.92 % | +10.55 pp | 30.29 % | +13.92 pp | 27.49 % | +11.12 pp | 17.44 % | +1.07 pp |
| Memory Throughput (%) | 41.22 % | 20.19 % | -21.03 pp | 21.10 % | -20.12 pp | 20.65 % | -20.57 pp | 40.60 % | -0.62 pp |
| DRAM Throughput (%) | 20.65 % | 4.01 % | -16.64 pp | 3.94 % | -16.71 pp | 3.83 % | -16.82 pp | 20.69 % | +0.04 pp |
| Achieved Occupancy (%) | 55.11 % | 69.19 % | +14.08 pp | 69.07 % | +13.96 pp | 73.88 % | +18.77 pp | 56.53 % | +1.42 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.71 % | 38.20 % | -32.51 pp | 33.42 % | -37.29 pp | 41.09 % | -29.62 pp | 69.97 % | -0.74 pp |
| IPC Active | 1.16 inst/cycle | 2.46 inst/cycle | +1.30 | 2.55 inst/cycle | +1.39 | 2.05 inst/cycle | +0.89 | 1.17 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan C: `revert_persistent_loop_to_2d_grid_with_autotune` (1.417×)

Reducing the autotune config set from 20 configs to 8 configs (while keeping the same 2D grid structure and num_stages=3) produced a 1.42x speedup. The NCU-measured per-kernel duration was nearly identical before and after (4.19 µs vs 4.22 µs), confirming the speedup came from reduced autotune search overhead at benchmark time rather than per-invocation latency improvement. The key metrics that did shift slightly were warp cycles per issued instruction (20.23→16.95) and eligible warps per scheduler (4.05→4.67), suggesting the trimmed config set allowed the autotuner to converge on a marginally better tile configuration faster. The 2D grid structure (no while-loop) was preserved throughout, keeping registers at 18/thread and theoretical occupancy at 100%.

> **Applicable when**: Triton kernel already uses @triton.autotune with a large config set (>10 configs) and the benchmark measures wall-clock time including autotune search; the per-kernel NCU duration is already near-optimal; reducing config count does not remove the globally optimal config from the search space.

> **Avoid when**: The kernel has not yet been autotuned and the optimal config is unknown — trimming too aggressively may exclude the best config; the benchmark uses cached autotune results (autotune overhead is paid only once); the kernel duration varies significantly across problem sizes and a wide config sweep is needed for generalization.

---

## Iteration 5

**Best so far**: 1.4479× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_autotuned_meta_parameters | 1.455× | 1.473× | ✓ |  |
| B | fix_broken_autotune_grid_launcher_and_reduce_num_stages | 1.253× | 1.353× | ✓ |  |
| C | fix_broken_autotune_grid_lambda_to_enable_tile_exploration | 1.397× | 1.384× | ✓ |  |
| D | shared_memory_staging_transpose_for_coalesced_scatter_store | 1.454× | 1.365× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_broken_autotu... | Δ | fix_broken_autotu... | Δ | shared_memory_sta... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4060 ns | -1920 ns (-32.1%) | 4420 ns | -1560 ns (-26.1%) | 4320 ns | -1660 ns (-27.8%) | 4380 ns | -1600 ns (-26.8%) |
| SM Throughput (%) | 16.37 % | 27.25 % | +10.88 pp | 29.72 % | +13.35 pp | 26.97 % | +10.60 pp | 25.59 % | +9.22 pp |
| Memory Throughput (%) | 41.22 % | 20.46 % | -20.76 pp | 20.72 % | -20.50 pp | 20.28 % | -20.94 pp | 19.20 % | -22.02 pp |
| DRAM Throughput (%) | 20.65 % | 4.07 % | -16.58 pp | 3.74 % | -16.91 pp | 3.83 % | -16.82 pp | 3.77 % | -16.88 pp |
| Achieved Occupancy (%) | 55.11 % | 68.64 % | +13.53 pp | 69.29 % | +14.18 pp | 70.24 % | +15.13 pp | 68.66 % | +13.55 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.71 % | 36.54 % | -34.17 pp | 36.38 % | -34.33 pp | 36.53 % | -34.18 pp | 37.09 % | -33.62 pp |
| IPC Active | 1.16 inst/cycle | 2.43 inst/cycle | +1.27 | 2.54 inst/cycle | +1.38 | 2.45 inst/cycle | +1.29 | 2.39 inst/cycle | +1.23 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_use_autotuned_meta_parameters` (1.473×)

When a Triton kernel's launch site uses a hardcoded grid (e.g., `(cdiv(xnumel, 128), cdiv(ynumel, 8))`) instead of a `lambda meta:` grid, the @triton.autotune decorator is effectively broken — every config runs with the same fixed tile size regardless of what the autotuner selects. Fixing the grid to `lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK']))` makes autotune functional, allowing the runtime to actually explore different tile configurations. The NCU-measured per-kernel duration improvement was modest (4.13 µs → 4.06 µs, ~1.7%), but the wall-clock speedup of 1.47x reflects the autotuner now converging on a better config and/or reduced overhead from the corrected config search. Additionally, removing the `tl.max_contiguous`/`tl.multiple_of` contiguity hint from xindex (reverting to plain `tl.arange`) and adjusting num_warps on some configs contributed to the config space refinement.

> **Applicable when**: Triton kernel uses @triton.autotune with XBLOCK/YBLOCK configs but the launch site grid is hardcoded to fixed constants rather than using `lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK']))`. NCU LaunchConfiguration rule is triggered (partial wave overhead). The autotune key includes dimensions that are effectively constant at runtime.

> **Avoid when**: The grid is already a lambda using meta parameters (autotune is already functional). The kernel intentionally uses a fixed grid for reasons unrelated to tile size (e.g., persistent kernels, cooperative groups). The problem size varies significantly across calls and the autotune key must include all variable dimensions.

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     4350 ns  (1.375×)  — fix_uncoalesced_stores_via_autotune_tile_reshape
Iter 2     4160 ns  (1.438×)  — autotune_yblock_aligned_to_stride_period
Iter 4     4130 ns  (1.448×)  — stride3_aligned_yblock_configs_with_vectorized_load_hints

Final best: 4130 ns  (1.4479×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +35,14 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
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
@@ -30,7 +50,7 @@

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
