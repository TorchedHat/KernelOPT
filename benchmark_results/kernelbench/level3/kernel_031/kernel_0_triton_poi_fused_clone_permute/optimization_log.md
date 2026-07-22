# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10750 ns — **Best latency**: 74980 ns (**0.1434×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_0_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 24
- **Duration**: 10750 ns
- **SM Throughput**: 15.49 % | **Memory**: 55.55 % | **DRAM**: 34.11 %
- **Achieved Occupancy**: 80.01 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.796× | 1.680× | ✓ |  |
| D | autotune_tile_sizes_for_coalescing_and_wave_efficiency | 1.898× | 1.940× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10750 ns | 6400 ns | -4350 ns (-40.5%) | 5540 ns | -5210 ns (-48.5%) |
| SM Throughput (%) | 15.49 % | 30.63 % | +15.14 pp | 37.79 % | +22.30 pp |
| Memory Throughput (%) | 55.55 % | 42.98 % | -12.57 pp | 32.29 % | -23.26 pp |
| DRAM Throughput (%) | 34.11 % | 26.76 % | -7.35 pp | 15.53 % | -18.58 pp |
| Achieved Occupancy (%) | 80.01 % | 84.12 % | +4.11 pp | 77.59 % | -2.42 pp |
| Registers/Thread | 24.00 register/thread | 23.00 register/thread | -1 | 25.00 register/thread | +1 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 80.07 % | 53.33 % | -26.74 pp | 35.33 % | -44.74 pp |
| IPC Active | 0.79 inst/cycle | 1.87 inst/cycle | +1.08 | 2.62 inst/cycle | +1.83 |

---

## Iteration 2

**Best so far**: 1.9404× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_callsite_and_add_vectorized_load_hints | 1.875× | 1.930× | ✓ |  |
| B | eliminate_partial_wave_via_sm_aligned_grid_configs | 1.824× | 1.954× | ✓ |  |
| C | fix_autotune_grid_lambda_and_expand_coalescing_configs | 1.919× | 1.680× | ✓ | **✓ best** |
| D | fix_autotune_grid_lambda_and_eliminate_partial_wave | 1.809× | 1.680× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_callsite... | Δ | eliminate_partial... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10750 ns | 5570 ns | -5180 ns (-48.2%) | 5500 ns | -5250 ns (-48.8%) | 6400 ns | -4350 ns (-40.5%) | 6400 ns | -4350 ns (-40.5%) |
| SM Throughput (%) | 15.49 % | 37.42 % | +21.93 pp | 37.33 % | +21.84 pp | 29.45 % | +13.96 pp | 30.39 % | +14.90 pp |
| Memory Throughput (%) | 55.55 % | 32.05 % | -23.50 pp | 31.97 % | -23.58 pp | 41.54 % | -14.01 pp | 42.93 % | -12.62 pp |
| DRAM Throughput (%) | 34.11 % | 15.53 % | -18.58 pp | 15.57 % | -18.54 pp | 26.85 % | -7.26 pp | 26.82 % | -7.29 pp |
| Achieved Occupancy (%) | 80.01 % | 75.69 % | -4.32 pp | 75.48 % | -4.53 pp | 84.38 % | +4.37 pp | 84.39 % | +4.38 pp |
| Registers/Thread | 24.00 register/thread | 25.00 register/thread | +1 | 25.00 register/thread | +1 | 23.00 register/thread | -1 | 23.00 register/thread | -1 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 80.07 % | 36.56 % | -43.51 pp | 38.55 % | -41.52 pp | 52.56 % | -27.51 pp | 52.99 % | -27.08 pp |
| IPC Active | 0.79 inst/cycle | 2.47 inst/cycle | +1.68 | 2.49 inst/cycle | +1.70 | 1.88 inst/cycle | +1.09 | 1.86 inst/cycle | +1.07 |


### Key Learnings

#### ✅ Plan C: `fix_autotune_grid_lambda_and_expand_coalescing_configs` (1.857×)

When a Triton kernel uses @triton.autotune but the call site passes a hardcoded grid tuple instead of a lambda reading META['XBLOCK']/META['YBLOCK'], autotune is effectively a no-op and the kernel always runs with the hardcoded tile shape. Replacing the hardcoded grid with `grid = lambda META: (cdiv(N, META['XBLOCK']), cdiv(M, META['YBLOCK']))` and removing explicit XBLOCK/YBLOCK kwargs from the call site allows autotune to actually select the best tile configuration at runtime. The NCU-measured single-kernel metrics show minimal change (the profiler captures a specific config run), but the wall-clock speedup of 1.86x reflects the autotuner discovering a better tile shape; pairing this with coalescing-friendly configs (large YBLOCK when Y is stride-1 in the load address) and tl.max_contiguous/tl.multiple_of vectorization hints further enables the compiler to generate efficient memory access code.

> **Applicable when**: Triton kernel uses @triton.autotune with tile-size configs (XBLOCK/YBLOCK) but the call site uses a hardcoded grid tuple instead of a lambda reading META; LaunchConfiguration NCU rule fires with partial-wave fraction ≥30%; load address has large stride in X dimension (non-coalesced) and stride-1 in Y dimension; L1 hit rate is 0% indicating no cache reuse; wall-clock speedup is the primary signal since NCU captures a single config run.

> **Avoid when**: The call site already uses a lambda grid (autotune is already functional); do not add large-YBLOCK configs when Y is the high-stride (non-contiguous) dimension in the load address; do not use tl.trans() or shared memory reordering on this transpose pattern (prior attempts failed correctness checks); kernel is already compute-bound (Compute SM Throughput >80%).

---

## Iteration 3

**Best so far**: 1.9404× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_and_add_sm_aligned_configs | 1.916× | 1.857× | ✓ |  |
| B | fix_static_grid_to_lambda_grid_enabling_autotune | 1.846× | 1.654× | ✓ |  |
| C | streaming_eviction_policy_with_lambda_grid_fix | 1.906× | 1.940× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10750 ns | 5790 ns | -4960 ns (-46.1%) | 6500 ns | -4250 ns (-39.5%) | 5540 ns | -5210 ns (-48.5%) |
| SM Throughput (%) | 15.49 % | 37.48 % | +21.99 pp | 30.27 % | +14.78 pp | 35.99 % | +20.50 pp |
| Memory Throughput (%) | 55.55 % | 32.06 % | -23.49 pp | 42.73 % | -12.82 pp | 30.71 % | -24.84 pp |
| DRAM Throughput (%) | 34.11 % | 14.83 % | -19.28 pp | 26.49 % | -7.62 pp | 15.54 % | -18.57 pp |
| Achieved Occupancy (%) | 80.01 % | 77.38 % | -2.63 pp | 84.32 % | +4.31 pp | 75.45 % | -4.56 pp |
| Registers/Thread | 24.00 register/thread | 25.00 register/thread | +1 | 23.00 register/thread | -1 | 25.00 register/thread | +1 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 80.07 % | 36.24 % | -43.83 pp | 52.53 % | -27.54 pp | 36.10 % | -43.97 pp |
| IPC Active | 0.79 inst/cycle | 2.32 inst/cycle | +1.53 | 1.88 inst/cycle | +1.09 | 2.49 inst/cycle | +1.70 |

---

## Iteration 4

**Best so far**: 1.9404× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | full_x_dimension_coverage_with_xblock256_and_lambda_grid | 1.802× | 0.143× | ✓ |  |
| B | fix_static_grid_and_maximize_warp_count_for_latency_hiding | 1.794× | 1.672× | ✓ |  |
| C | fix_callsite_lambda_grid_with_full_xblock_coverage | 1.914× | 0.143× | ✓ |  |
| D | add_triton_autotune | 1.976× | 1.899× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | full_x_dimension_... | Δ | fix_static_grid_a... | Δ | fix_callsite_lamb... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10750 ns | 74910 ns | +64160 ns (+596.8%) | 6430 ns | -4320 ns (-40.2%) | 75010 ns | +64260 ns (+597.8%) | 5660 ns | -5090 ns (-47.3%) |
| SM Throughput (%) | 15.49 % | 69.02 % | +53.53 pp | 29.81 % | +14.32 pp | 69.12 % | +53.63 pp | 37.00 % | +21.51 pp |
| Memory Throughput (%) | 55.55 % | 70.48 % | +14.93 pp | 42.05 % | -13.50 pp | 70.58 % | +15.03 pp | 31.61 % | -23.94 pp |
| DRAM Throughput (%) | 34.11 % | 4.77 % | -29.34 pp | 26.61 % | -7.50 pp | 4.75 % | -29.36 pp | 15.23 % | -18.88 pp |
| Achieved Occupancy (%) | 80.01 % | 24.45 % | -55.56 pp | 84.32 % | +4.31 pp | 24.45 % | -55.56 pp | 75.68 % | -4.33 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 23.00 register/thread | -1 | 32.00 register/thread | +8 | 25.00 register/thread | +1 |
| Waves/SM | 1.94 | 31.03 | +29.09 | 1.94 | +0.00 | 31.03 | +29.09 | 1.94 | +0.00 |
| No Eligible (%) | 80.07 % | 51.05 % | -29.02 pp | 53.76 % | -26.31 pp | 50.99 % | -29.08 pp | 36.97 % | -43.10 pp |
| IPC Active | 0.79 inst/cycle | 1.93 inst/cycle | +1.14 | 1.86 inst/cycle | +1.07 | 1.93 inst/cycle | +1.14 | 2.50 inst/cycle | +1.71 |

---

## Iteration 5

**Best so far**: 1.9404× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | unlock_autotune_via_lambda_grid_for_vectorized_loads | 1.980× | 0.143× | ✓ |  |
| B | fix_static_grid_callsite_to_lambda_and_evict_first_on_strided_load | 1.831× | 1.672× | ✓ |  |
| C | replace_smem_heavy_autotune_configs_with_smem_safe_configs_and_fix_lambda_grid | 1.856× | 0.547× | ✓ |  |
| D | eliminate_shared_memory_to_restore_occupancy | 1.990× | 0.143× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | unlock_autotune_v... | Δ | fix_static_grid_c... | Δ | replace_smem_heav... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10750 ns | 74940 ns | +64190 ns (+597.1%) | 6430 ns | -4320 ns (-40.2%) | 19650 ns | +8900 ns (+82.8%) | 74980 ns | +64230 ns (+597.5%) |
| SM Throughput (%) | 15.49 % | 69.15 % | +53.66 pp | 29.87 % | +14.38 pp | 41.06 % | +25.57 pp | 69.11 % | +53.62 pp |
| Memory Throughput (%) | 55.55 % | 70.61 % | +15.06 pp | 42.22 % | -13.33 pp | 77.51 % | +21.96 pp | 70.58 % | +15.03 pp |
| DRAM Throughput (%) | 34.11 % | 4.76 % | -29.35 pp | 26.78 % | -7.33 pp | 18.13 % | -15.98 pp | 4.75 % | -29.36 pp |
| Achieved Occupancy (%) | 80.01 % | 24.46 % | -55.55 pp | 84.21 % | +4.20 pp | 67.23 % | -12.78 pp | 24.46 % | -55.55 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 23.00 register/thread | -1 | 39.00 register/thread | +15 | 32.00 register/thread | +8 |
| Waves/SM | 1.94 | 31.03 | +29.09 | 1.94 | +0.00 | 5.17 | +3.23 | 31.03 | +29.09 |
| No Eligible (%) | 80.07 % | 51.27 % | -28.80 pp | 53.24 % | -26.83 pp | 53.51 % | -26.56 pp | 51.04 % | -29.03 pp |
| IPC Active | 0.79 inst/cycle | 1.93 inst/cycle | +1.14 | 1.85 inst/cycle | +1.06 | 1.86 inst/cycle | +1.07 | 1.93 inst/cycle | +1.14 |

---

## Cumulative Progress

```
Baseline:  10750 ns  (1.000×)
Iter 1     5540 ns  (1.940×)  — autotune_tile_sizes_for_coalescing_and_wave_efficiency

Final best: 74980 ns  (0.1434×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,48 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 128}, num_warps=16),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=16),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 2,   'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 256}, num_warps=16),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_permute_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
     xnumel = 256
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    y_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK) + yoffset, YBLOCK), YBLOCK)
+    yindex = y_arange[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    x_arange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK) + xoffset, XBLOCK), XBLOCK)
+    xindex = x_arange[None, :]
     xmask = xindex < xnumel
     x1 = xindex
     y0 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 16384*x1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 16384*x1), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -28,7 +54,7 @@

     ynumel = 16384
     xnumel = 256
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_clone_permute_view_0[grid](arg_0, arg_1, 16384, 256, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_clone_permute_view_0[grid](arg_0, arg_1, 16384, 256)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
