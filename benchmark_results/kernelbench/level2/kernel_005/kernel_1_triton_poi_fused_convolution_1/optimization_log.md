# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285600 ns — **Best latency**: 140830 ns (**2.0280×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_add_streaming_store_cache_modifier`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_005/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285600 ns
- **SM Throughput**: 19.38 % | **Memory**: 78.91 % | **DRAM**: 72.24 %
- **Achieved Occupancy**: 94.06 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_with_autotune | 2.343× | 2.012× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285600 ns | 141920 ns | -143680 ns (-50.3%) |
| SM Throughput (%) | 19.38 % | 47.81 % | +28.43 pp |
| Memory Throughput (%) | 78.91 % | 41.73 % | -37.18 pp |
| DRAM Throughput (%) | 72.24 % | 34.67 % | -37.57 pp |
| Achieved Occupancy (%) | 94.06 % | 89.12 % | -4.94 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 56.39 % | -23.92 pp |
| IPC Active | 0.79 inst/cycle | 1.74 inst/cycle | +0.95 |

---

## Iteration 2

**Best so far**: 2.0124× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_for_coalesced_stores_and_latency_hiding | 2.342× | 1.994× | ✓ |  |
| D | fix_autotune_grid_lambda_and_expand_tile_configs | 2.342× | 2.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285600 ns | 143260 ns | -142340 ns (-49.8%) | 141820 ns | -143780 ns (-50.3%) |
| SM Throughput (%) | 19.38 % | 46.54 % | +27.16 pp | 47.64 % | +28.26 pp |
| Memory Throughput (%) | 78.91 % | 40.69 % | -38.22 pp | 41.49 % | -37.42 pp |
| DRAM Throughput (%) | 72.24 % | 34.32 % | -37.92 pp | 34.72 % | -37.52 pp |
| Achieved Occupancy (%) | 94.06 % | 90.02 % | -4.04 pp | 86.65 % | -7.41 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 56.68 % | -23.63 pp | 56.87 % | -23.44 pp |
| IPC Active | 0.79 inst/cycle | 1.76 inst/cycle | +0.97 | 1.71 inst/cycle | +0.92 |


### Key Learnings

#### ❌ Plan A: `expand_autotune_configs_for_coalesced_stores_and_latency_hiding` (1.985×)

Applying tl.max_contiguous/tl.multiple_of hints to the store-side index and expanding the autotune config set with additional num_stages=5/6 configs caused kernel duration to increase from 140.83us to 143.9us despite marginal improvements in eligible warps per scheduler (2.63→3.29) and achieved occupancy (89.07%→92.05%). The autotuner selected a config from the expanded set that is slightly worse than the prior winner, and the store vectorization hints did not measurably improve bytes-per-sector on stores (still 28.4/32). Once a transpose kernel's tile shape has been optimized via autotuning and the .cs cache modifier applied, adding store-side contiguity hints and deeper pipeline stage configs provides no further benefit and may cause the autotuner to select a suboptimal config.

> **Applicable when**: Triton transpose kernel where tile shape has already been autotuned (L2 ~47%, warp stall cycles ~32/instruction, IssueSlotUtilization still flagged), and the plan is to add store-side tl.max_contiguous/tl.multiple_of hints plus num_stages=5/6 autotune configs to further reduce the 28.4/32 bytes-per-sector store inefficiency.

> **Avoid when**: The kernel's tile shape has already been autotuned and duration is near its floor (~141us for this problem size). Adding store-side vectorization hints and deeper pipeline stage configs to an already-optimized transpose kernel causes the autotuner to select a slightly worse config, increasing duration by ~2%. Do not apply when the existing autotune winner already achieves near-optimal performance for this memory-bound transpose pattern.

---

## Iteration 3

**Best so far**: 2.0124× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_software_pipeline_latency_hiding | 2.320× | 1.964× | ✓ |  |
| B | fix_grid_lambda_and_add_streaming_store_cache_modifier | 2.349× | 2.028× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285600 ns | 145410 ns | -140190 ns (-49.1%) | 140830 ns | -144770 ns (-50.7%) |
| SM Throughput (%) | 19.38 % | 46.20 % | +26.82 pp | 47.78 % | +28.40 pp |
| Memory Throughput (%) | 78.91 % | 40.07 % | -38.84 pp | 41.60 % | -37.31 pp |
| DRAM Throughput (%) | 72.24 % | 33.19 % | -39.05 pp | 34.52 % | -37.72 pp |
| Achieved Occupancy (%) | 94.06 % | 91.99 % | -2.07 pp | 89.07 % | -4.99 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 55.14 % | -25.17 pp | 56.91 % | -23.40 pp |
| IPC Active | 0.79 inst/cycle | 1.79 inst/cycle | +1.00 | 1.74 inst/cycle | +0.95 |

---

## Iteration 4

**Best so far**: 2.028× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_store_hints_with_fixed_grid_lambda | 2.253× | 1.985× | ✓ |  |
| B | fix_grid_lambda_to_use_autotuned_blocks_and_swap_tile_axes | 2.319× | 1.944× | ✓ |  |
| C | streaming_store_cache_modifier_cs_write_once_output | 2.252× | 1.968× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_store_... | Δ | fix_grid_lambda_t... | Δ | streaming_store_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285600 ns | 143900 ns | -141700 ns (-49.6%) | 146880 ns | -138720 ns (-48.6%) | 145090 ns | -140510 ns (-49.2%) |
| SM Throughput (%) | 19.38 % | 46.94 % | +27.56 pp | 46.28 % | +26.90 pp | 47.53 % | +28.15 pp |
| Memory Throughput (%) | 78.91 % | 40.88 % | -38.03 pp | 40.20 % | -38.71 pp | 41.37 % | -37.54 pp |
| DRAM Throughput (%) | 72.24 % | 33.88 % | -38.36 pp | 32.87 % | -39.37 pp | 33.64 % | -38.60 pp |
| Achieved Occupancy (%) | 94.06 % | 92.05 % | -2.01 pp | 92.00 % | -2.06 pp | 91.91 % | -2.15 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 54.21 % | -26.10 pp | 55.60 % | -24.71 pp | 53.94 % | -26.37 pp |
| IPC Active | 0.79 inst/cycle | 1.83 inst/cycle | +1.04 | 1.80 inst/cycle | +1.01 | 1.83 inst/cycle | +1.04 |

---

## Iteration 5

**Best so far**: 2.028× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_corrected_autotune_grid | 2.318× | 1.962× | ✓ |  |
| B | vectorized_store_hints_isolated_xblock_multiple_of | 2.316× | 1.956× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_store_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285600 ns | 145600 ns | -140000 ns (-49.0%) | 146020 ns | -139580 ns (-48.9%) |
| SM Throughput (%) | 19.38 % | 46.53 % | +27.15 pp | 46.73 % | +27.35 pp |
| Memory Throughput (%) | 78.91 % | 40.38 % | -38.53 pp | 40.54 % | -38.37 pp |
| DRAM Throughput (%) | 72.24 % | 33.15 % | -39.09 pp | 33.01 % | -39.23 pp |
| Achieved Occupancy (%) | 94.06 % | 91.97 % | -2.09 pp | 91.93 % | -2.13 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.31 % | 54.90 % | -25.41 pp | 54.86 % | -25.45 pp |
| IPC Active | 0.79 inst/cycle | 1.78 inst/cycle | +0.99 | 1.82 inst/cycle | +1.03 |


### Key Learnings

#### ❌ Plan B: `vectorized_store_hints_isolated_xblock_multiple_of` (1.956×)

Inserting `tl.max_contiguous(tl.multiple_of(xindex, [1, 16]), [1, XBLOCK])` on the xindex immediately after assignment (before the store) caused kernel duration to increase from 143.9us to 146.02us. All key metrics either stayed flat or slightly worsened: No Eligible warp cycles increased from 54.21% to 54.86%, eligible warps per scheduler dropped from 3.29 to 3.26, and warp cycles per issued instruction rose from 32.38 to 32.79. The bytes-per-sector on stores remained at 28.4/32 (MemoryCacheAccessPattern rule still triggered at the same severity), confirming the hint did not enable 128-bit vectorized stores. For this transpose kernel, the store-side contiguity hint on xindex provides no measurable benefit and slightly regresses performance, likely because the compiler already infers sufficient coalescing from the stride-1 store pattern, and the hint introduces minor overhead.

> **Applicable when**: Triton transpose kernel where MemoryCacheAccessPattern reports 28.4/32 bytes per sector on stores, store address has stride-1 in the fast (warp-contiguous) dimension, and the plan is to add tl.max_contiguous/tl.multiple_of hints to xindex to enable 128-bit vectorized stores.

> **Avoid when**: The kernel is a strided transpose where the store side is already stride-1 contiguous (x2 + constant*y3 pattern). The compiler already infers coalescing for stride-1 stores; adding tl.max_contiguous/tl.multiple_of hints to xindex does not improve bytes-per-sector (remains 28.4/32) and slightly increases kernel duration (~1.5%). Also note: this iteration also removed cache_modifier='.cs' from the store, which may have contributed to the regression — do not remove .cs when it was previously shown to help (see exp_003).

---

## Cumulative Progress

```
Baseline:  285600 ns  (1.000×)
Iter 1     141920 ns  (2.012×)  — shared_memory_tiled_transpose_with_autotune
Iter 3     140830 ns  (2.028×)  — fix_grid_lambda_and_add_streaming_store_cache_modifier

Final best: 140830 ns  (2.0280×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=6),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=6),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=6),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +43,9 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_first')
+    x2_vec = tl.max_contiguous(tl.multiple_of(x2, [1, XBLOCK]), [1, XBLOCK])
+    tl.store(out_ptr0 + (x2_vec + 31500*y3), tmp0, xmask & ymask, cache_modifier='.cs')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +53,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
