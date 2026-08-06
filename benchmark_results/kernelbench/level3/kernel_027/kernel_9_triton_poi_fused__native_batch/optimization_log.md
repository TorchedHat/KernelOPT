# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5180 ns — **Best latency**: 4480 ns (**1.1562×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`persistent_tile_scheduling_fix_static_grid_to_meta_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 28
- **Duration**: 5180 ns
- **SM Throughput**: 40.56 % | **Memory**: 40.01 % | **DRAM**: 4.68 %
- **Achieved Occupancy**: 70.75 %
- **IPC Active**: 2.78 inst/cycle | **No Eligible**: 29.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_shape_to_eliminate_partial_wave_and_improve_coalescing | 1.071× | 1.164× | ✓ | **✓ best** |
| D | fix_tile_dimensions_and_vectorize_loads | 1.070× | 1.164× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_shape_to... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 4450 ns | -730 ns (-14.1%) | 4450 ns | -730 ns (-14.1%) |
| SM Throughput (%) | 40.56 % | 12.99 % | -27.57 pp | 15.74 % | -24.82 pp |
| Memory Throughput (%) | 40.01 % | 5.47 % | -34.54 pp | 6.13 % | -33.88 pp |
| DRAM Throughput (%) | 4.68 % | 5.47 % | +0.79 pp | 5.44 % | +0.76 pp |
| Achieved Occupancy (%) | 70.75 % | 28.66 % | -42.09 pp | 31.21 % | -39.54 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 29.47 % | 65.98 % | +36.51 pp | 54.62 % | +25.15 pp |
| IPC Active | 2.78 inst/cycle | 1.28 inst/cycle | -1.50 | 1.60 inst/cycle | -1.18 |

---

## Iteration 2

**Best so far**: 1.164× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.005× | 0.003× | ✓ |  |
| B | fix_autotune_grid_lambda_and_expand_yblock_configs | 1.053× | 1.172× | ✓ |  |
| C | persistent_tile_scheduling_fix_static_grid_to_meta_lambda | 1.073× | 1.156× | ✓ | **✓ best** |
| D | fix_autotune_grid_to_use_meta_block_sizes | 1.062× | 1.164× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 1610000 ns | +1604820 ns (+30981.1%) | 4420 ns | -760 ns (-14.7%) | 4480 ns | -700 ns (-13.5%) | 4450 ns | -730 ns (-14.1%) |
| SM Throughput (%) | 40.56 % | 76.46 % | +35.90 pp | 12.66 % | -27.90 pp | 16.07 % | -24.49 pp | 15.82 % | -24.74 pp |
| Memory Throughput (%) | 40.01 % | 69.00 % | +28.99 pp | 5.46 % | -34.55 pp | 6.13 % | -33.88 pp | 6.16 % | -33.85 pp |
| DRAM Throughput (%) | 4.68 % | 0.02 % | -4.66 pp | 5.46 % | +0.78 pp | 5.44 % | +0.76 pp | 5.45 % | +0.77 pp |
| Achieved Occupancy (%) | 70.75 % | 80.13 % | +9.38 pp | 28.36 % | -42.39 pp | 39.52 % | -31.23 pp | 33.78 % | -36.97 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 29.47 % | 25.50 % | -3.97 pp | 62.29 % | +32.82 pp | 52.86 % | +23.39 pp | 54.26 % | +24.79 pp |
| IPC Active | 2.78 inst/cycle | 2.98 inst/cycle | +0.20 | 1.30 inst/cycle | -1.48 | 1.39 inst/cycle | -1.39 | 1.54 inst/cycle | -1.24 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.003×)

Converting a short-lived 2D grid kernel into a persistent 1D grid with an inner while-loop caused a catastrophic regression: executed instructions exploded from 429K to 944M (~2200x), and duration increased from 4.42µs to 1.62ms (~366x slower). Although occupancy improved (28.4% → 79.9%) and L1/L2 hit rates rose dramatically (L2: 20% → 99%), these gains were completely overwhelmed by the instruction count explosion. The root cause is that the persistent while-loop forces each SM to iterate through many tiles sequentially, and the per-iteration arithmetic (modulo, integer division for tile decomposition) generates enormous instruction overhead that dwarfs the actual memory work. This iteration also removed 4 of the 8 autotune configs (YBLOCK=2048, 4096 and XBLOCK=16 variants), further limiting the search space. For extremely short kernels (duration < 10µs, total work < 1M instructions), persistent scheduling adds more overhead than it saves from eliminating partial waves.

> **Applicable when**: Never applicable for kernels with very low total instruction counts (<1M instructions, duration <10µs). The while-loop overhead (modulo, integer division, multiply for tile decomposition + loop branch) generates more instructions than the actual memory work being done per tile.

> **Avoid when**: Any kernel where the total work per CTA is very small (e.g., a simple layout permutation with 128 elements per block). Persistent scheduling is only beneficial when each tile does substantial compute (e.g., matmul tiles, large reductions) where the loop overhead is negligible relative to the tile work. Never combine persistent scheduling with autotune config reduction — both changes compound the regression.

---

## Iteration 3

**Best so far**: 1.164× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_meta_lambda_to_activate_autotune | 0.005× | 0.003× | ✓ |  |
| B | fix_grid_lambda_to_honor_autotune_meta_block_sizes | 1.042× | 1.131× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 1620000 ns | +1614820 ns (+31174.1%) | 4580 ns | -600 ns (-11.6%) |
| SM Throughput (%) | 40.56 % | 76.74 % | +36.18 pp | 15.78 % | -24.78 pp |
| Memory Throughput (%) | 40.01 % | 68.90 % | +28.89 pp | 6.13 % | -33.88 pp |
| DRAM Throughput (%) | 4.68 % | 0.02 % | -4.66 pp | 5.32 % | +0.64 pp |
| Achieved Occupancy (%) | 70.75 % | 79.94 % | +9.19 pp | 30.74 % | -40.01 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 29.47 % | 24.77 % | -4.70 pp | 54.29 % | +24.82 pp |
| IPC Active | 2.78 inst/cycle | 2.99 inst/cycle | +0.21 | 1.62 inst/cycle | -1.16 |


### Key Learnings

#### ✅ Plan B: `fix_grid_lambda_to_honor_autotune_meta_block_sizes` (1.131×)

Changing the autotune config set (removing some XBLOCK=1 large-YBLOCK configs and adding XBLOCK=16 variants with num_stages=1) alongside a grid lambda that reads META['XBLOCK'] and META['YBLOCK'] produced a modest 1.13x speedup. However, the winning config increased executed instructions (429K → 542K) and elapsed cycles (6608 → 6783), while improving scheduler eligible warps (1.28 → 1.67), reducing warp cycles per issued instruction (13.05 → 11.13), and improving L2 hit rate (20.32% → 36.64%). The net speedup is real but smaller than predicted, suggesting the autotune selected an XBLOCK=16 config that trades instruction efficiency for better cache reuse. When the grid lambda is already correct and the bottleneck is scheduler idle cycles, trimming the config set to only XBLOCK=1 variants with large YBLOCK is safer than introducing XBLOCK=16 configs that increase instruction count.

> **Applicable when**: Triton kernel where the grid lambda already correctly reads META['XBLOCK'] and META['YBLOCK'], scheduler No Eligible is high (>50%), eligible warps per scheduler is low (<1.5), and L2 hit rate is low (<25%). Adding XBLOCK=16 configs with num_stages=1 can improve cache reuse and scheduler utilization when xnumel allows XBLOCK=16 without excessive masking.

> **Avoid when**: The grid is still a static tuple (fix the grid lambda first). The autotune already includes XBLOCK=16 configs. Executed instruction count is already low and the bottleneck is memory bandwidth rather than scheduler idle cycles. xnumel is very small (e.g., 9) and XBLOCK=16 would still mask 7/16 threads, wasting ~44% of x-threads.

---

## Iteration 4

**Best so far**: 1.164× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  5180 ns  (1.000×)
Iter 1     4450 ns  (1.164×)  — fix_tile_shape_to_eliminate_partial_wave_and_improve_coalescing

Final best: 4480 ns  (1.1562×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 1}, num_warps=4),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 1}, num_warps=4),
+        triton.Config({'YBLOCK': 512, 'XBLOCK': 1}, num_warps=8),
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=8),
+        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=8),
+        triton.Config({'YBLOCK': 4096, 'XBLOCK': 1}, num_warps=16),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 32768
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = (yoffset + tl.arange(0, YBLOCK)[:, None]) < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +33,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 32768
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9[grid](arg_0, arg_1, 32768, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9[grid](arg_0, arg_1, 32768, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
