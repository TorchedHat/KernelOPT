# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7100 ns — **Best latency**: 6500 ns (**1.0923×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`revert_broken_persistent_loop_to_2d_grid_with_autotune_and_correct_grid_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_228_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7296,1,1), Waves/SM: 3.45
- **Registers/thread**: 28
- **Duration**: 7100 ns
- **SM Throughput**: 54.99 % | **Memory**: 51.95 % | **DRAM**: 6.04 %
- **Achieved Occupancy**: 71.37 %
- **IPC Active**: 3.14 inst/cycle | **No Eligible**: 20.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_tile_dimensions_for_coalesced_access_and_full_waves | 0.982× | 0.867× | ✓ |  |
| D | vectorize_loads_stores_via_tile_reshape_and_autotune | 1.046× | 1.092× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_tile_dimensio... | Δ | vectorize_loads_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7100 ns | 8190 ns | +1090 ns (+15.4%) | 6500 ns | -600 ns (-8.5%) |
| SM Throughput (%) | 54.99 % | 60.94 % | +5.95 pp | 16.32 % | -38.67 pp |
| Memory Throughput (%) | 51.95 % | 42.50 % | -9.45 pp | 6.63 % | -45.32 pp |
| DRAM Throughput (%) | 6.04 % | 5.25 % | -0.79 pp | 6.63 % | +0.59 pp |
| Achieved Occupancy (%) | 71.37 % | 79.34 % | +7.97 pp | 20.70 % | -50.67 pp |
| Registers/Thread | 28.00 register/thread | 27.00 register/thread | -1 | 16.00 register/thread | -12 |
| Waves/SM | 3.45 | 3.45 | +0.00 | 3.45 | +0.00 |
| No Eligible (%) | 20.92 % | 21.65 % | +0.73 pp | 63.93 % | +43.01 pp |
| IPC Active | 3.14 inst/cycle | 3.16 inst/cycle | +0.02 | 1.24 inst/cycle | -1.90 |

---

## Iteration 2

**Best so far**: 1.0923× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_to_eliminate_partial_wave_and_workload_imbalance | 0.000× | 0.000× | ✓ |  |
| B | wire_autotune_grid_lambda_to_eliminate_fixed_tile_override | 1.043× | 1.092× | ✓ |  |
| C | persistent_tile_scheduling | 0.000× | 0.000× | ✓ |  |
| D | autotune_large_yblock_for_coalesced_transpose_stores_with_correct_grid_lambda | 1.044× | 1.052× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | wire_autotune_gri... | Δ | persistent_tile_s... | Δ | autotune_large_yb... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7100 ns | 36740000 ns | +36732900 ns (+517364.8%) | 6500 ns | -600 ns (-8.5%) | 36600000 ns | +36592900 ns (+515393.0%) | 6750 ns | -350 ns (-4.9%) |
| SM Throughput (%) | 54.99 % | 87.08 % | +32.09 pp | 16.31 % | -38.68 pp | 87.44 % | +32.45 pp | 31.99 % | -23.00 pp |
| Memory Throughput (%) | 51.95 % | 85.69 % | +33.74 pp | 6.63 % | -45.32 pp | 86.05 % | +34.10 pp | 9.58 % | -42.37 pp |
| DRAM Throughput (%) | 6.04 % | 0.01 % | -6.03 pp | 6.63 % | +0.59 pp | 0.01 % | -6.03 pp | 6.39 % | +0.35 pp |
| Achieved Occupancy (%) | 71.37 % | 89.66 % | +18.29 pp | 20.67 % | -50.70 pp | 89.77 % | +18.40 pp | 28.39 % | -42.98 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 3.45 | 3.45 | +0.00 | 3.45 | +0.00 | 3.45 | +0.00 | 6.91 | +3.46 |
| No Eligible (%) | 20.92 % | 11.78 % | -9.14 pp | 64.70 % | +43.78 pp | 11.44 % | -9.48 pp | 48.11 % | +27.19 pp |
| IPC Active | 3.14 inst/cycle | 3.53 inst/cycle | +0.39 | 1.23 inst/cycle | -1.91 | 3.54 inst/cycle | +0.40 | 1.95 inst/cycle | -1.19 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_to_eliminate_partial_wave_and_workload_imbalance` (0.001×)

Replacing a fixed 2D grid with a persistent while-loop kernel using tl.num_programs(0) as the tile stride caused a catastrophic ~2000x slowdown: duration went from 6.5 µs to 13.88 ms. Executed instructions exploded from 822K to 8.74 billion (+10,600x), indicating the while-loop iterated far more times than intended. The root cause is that the grid was launched as a 1D grid of size tl.num_programs(0) (the SM count), but the kernel body computed total_tiles using the constexpr XBLOCK/YBLOCK values from the autotune decorator — however, the autotune was still active and the grid lambda was not updated to match the persistent design, so the actual num_programs seen at runtime was wrong. This is a repeat of the same anti-pattern documented in exp_002: persistent while-loop kernels require exact coordination between the grid launch size, the XBLOCK/YBLOCK values, and the tile decomposition math, and any mismatch causes near-infinite looping.

> **Applicable when**: Never: persistent while-loop scheduling should not be applied to this kernel class (short-duration, small xnumel, autotune-decorated). The pattern is fundamentally incompatible with @triton.autotune because the grid size and XBLOCK/YBLOCK must be fixed at design time for the loop math to be correct.

> **Avoid when**: Always avoid combining persistent while-loop tile scheduling with @triton.autotune in Triton. The autotune selects XBLOCK/YBLOCK at runtime, but the persistent kernel's while-loop termination condition (total_tiles) and stride (num_programs) depend on these values being consistent with the actual grid launch size — a constraint that cannot be satisfied when autotune varies tile sizes. Also avoid when the kernel duration is <100 µs, as the while-loop overhead per tile dominates. This is the third consecutive regression from this direction (exp_002 and exp_003).

#### ❌ Plan C: `persistent_tile_scheduling` (0.000×)

Adding a persistent for-loop tile scheduler inside a Triton kernel while leaving the grid launch unchanged caused a catastrophic ~5600x regression: duration went from 6.5 µs to 36.59 ms and executed instructions exploded from 822K to 25.3 billion. The root cause is that the grid in kernel_function was never updated from the old fixed 2D tuple (cdiv(9,128), cdiv(58368,8)) = (1, 7296) to the intended persistent 1D grid (132,), so tl.num_programs(0)=1 and the for-loop iterated over all total_tiles per CTA instead of total_tiles/NUM_SMS. Additionally, tl.cdiv(runtime_arg, CONSTEXPR) inside the kernel body is not constant-folded, generating a massive dynamic instruction stream. Registers per thread doubled (16→32) and dynamic shared memory appeared (0→4 KB/block), confirming the JIT compiled a much heavier kernel than intended.

> **Applicable when**: Never: this is a pure anti-pattern. Do not apply persistent tile scheduling with a for-loop over tl.cdiv-computed bounds without simultaneously updating the grid to (NUM_SMS,).

> **Avoid when**: Always avoid this pattern. Specifically: (1) do not add a persistent for-loop inside a Triton kernel without updating the grid launch to a 1D grid of NUM_SMS; (2) do not use tl.cdiv(runtime_arg, CONSTEXPR) to compute loop bounds inside a Triton kernel — it is not constant-folded and generates a large dynamic instruction stream; (3) do not pass YBLOCK/XBLOCK values at the call site that differ from those assumed in the persistent tile decomposition math inside the kernel body.

---

## Iteration 3

**Best so far**: 1.0923× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_to_activate_tile_configs | 0.002× | 0.001× | ✓ |  |
| B | fix_grid_lambda_to_activate_existing_autotune_configs | 0.002× | 0.001× | ✓ |  |
| C | fix_persistent_loop_grid_to_activate_sm_level_tile_distribution | 0.000× | 0.000× | ✓ |  |
| D | revert_broken_persistent_loop_to_2d_grid_with_autotune_and_correct_grid_lambda | 1.043× | 1.092× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | fix_grid_lambda_t... | Δ | fix_persistent_lo... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7100 ns | 13880000 ns | +13872900 ns (+195393.0%) | 13880000 ns | +13872900 ns (+195393.0%) | 36590000 ns | +36582900 ns (+515252.1%) | 6500 ns | -600 ns (-8.5%) |
| SM Throughput (%) | 54.99 % | 79.52 % | +24.53 pp | 79.52 % | +24.53 pp | 87.45 % | +32.46 pp | 16.83 % | -38.16 pp |
| Memory Throughput (%) | 51.95 % | 29.54 % | -22.41 pp | 29.54 % | -22.41 pp | 86.06 % | +34.11 pp | 6.62 % | -45.33 pp |
| DRAM Throughput (%) | 6.04 % | 0.01 % | -6.03 pp | 0.01 % | -6.03 pp | 0.01 % | -6.03 pp | 6.62 % | +0.58 pp |
| Achieved Occupancy (%) | 71.37 % | 88.50 % | +17.13 pp | 88.50 % | +17.13 pp | 89.77 % | +18.40 pp | 20.57 % | -50.80 pp |
| Registers/Thread | 28.00 register/thread | 20.00 register/thread | -8 | 20.00 register/thread | -8 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 3.45 | 3.45 | +0.00 | 3.45 | +0.00 | 3.45 | +0.00 | 3.45 | +0.00 |
| No Eligible (%) | 20.92 % | 19.44 % | -1.48 pp | 19.43 % | -1.49 pp | 11.42 % | -9.50 pp | 64.05 % | +43.13 pp |
| IPC Active | 3.14 inst/cycle | 3.22 inst/cycle | +0.08 | 3.22 inst/cycle | +0.08 | 3.54 inst/cycle | +0.40 | 1.24 inst/cycle | -1.90 |

---

## Cumulative Progress

```
Baseline:  7100 ns  (1.000×)
Iter 1     6500 ns  (1.092×)  — vectorize_loads_stores_via_tile_reshape_and_autotune

Final best: 6500 ns  (1.0923×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 58368
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +33,8 @@

     y3 = yindex
     y0 = (yindex % 1824)
     y1 = yindex // 1824
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 1824*x2 + 16416*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 1824*x2 + 16416*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

     ynumel = 58368
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228[grid](arg_0, arg_1, 58368, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228[grid](arg_0, arg_1, 58368, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
