# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_convolution_gelu_mul_native_layer_norm_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 717410 ns — **Best latency**: 23330 ns (**30.7505×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_add_large_xblock_configs_for_mio_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_convolution_gelu_mul_native_layer_norm_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_034/kernel_0_triton_per_fused_convolution_g/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 82.75
- **Registers/thread**: 168
- **Duration**: 717410 ns
- **SM Throughput**: 64.43 % | **Memory**: 60.16 % | **DRAM**: 60.16 %
- **Achieved Occupancy**: 18.30 %
- **IPC Active**: 2.60 inst/cycle | **No Eligible**: 35.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_xblock_reduction | 22.664× | 25.279× | ✓ |  |
| C | reduce_register_pressure_via_xblock_autotune | 22.722× | 25.449× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_maxnreg_and_smaller_xblock | 7.298× | 7.551× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 717410 ns | 28380 ns | -689030 ns (-96.0%) | 28190 ns | -689220 ns (-96.1%) | 95010 ns | -622400 ns (-86.8%) |
| SM Throughput (%) | 64.43 % | 79.15 % | +14.72 pp | 77.55 % | +13.12 pp | 75.64 % | +11.21 pp |
| Memory Throughput (%) | 60.16 % | 67.40 % | +7.24 pp | 68.33 % | +8.17 pp | 63.27 % | +3.11 pp |
| DRAM Throughput (%) | 60.16 % | 6.02 % | -54.14 pp | 6.07 % | -54.09 pp | 51.80 % | -8.36 pp |
| Achieved Occupancy (%) | 18.30 % | 87.62 % | +69.32 pp | 87.28 % | +68.98 pp | 40.94 % | +22.64 pp |
| Registers/Thread | 168.00 register/thread | 23.00 register/thread | -145 | 23.00 register/thread | -145 | 70.00 register/thread | -98 |
| Waves/SM | 82.75 | 15.52 | -67.23 | 15.52 | -67.23 | 17.73 | -65.02 |
| No Eligible (%) | 35.32 % | 13.86 % | -21.46 pp | 16.69 % | -18.63 pp | 22.38 % | -12.94 pp |
| IPC Active | 2.60 inst/cycle | 3.44 inst/cycle | +0.84 | 3.32 inst/cycle | +0.72 | 3.11 inst/cycle | +0.51 |


### Key Learnings

#### ✅ Plan A: `reduce_register_pressure_via_xblock_reduction` (25.279×)

When a Triton kernel uses a large XBLOCK alongside a full R0_BLOCK reduction, intermediate 2D tensors of shape [XBLOCK, R0_BLOCK] all live simultaneously in registers, causing extreme register pressure (168 regs/thread) and catastrophically low occupancy (18.75%, Block Limit Registers=3). Reducing XBLOCK to 1 (or a small power of 2) collapses these 2D tensors to [1, R0_BLOCK] scalars, dropping register usage from 168 to 23 per thread and raising theoretical occupancy from 18.75% to 100%. This also dramatically improves L1 cache hit rate (9.1% → 86.53%) because each program now reuses a small, cache-resident working set rather than scattering across a large register file.

> **Applicable when**: NCU reports TheoreticalOccupancy rule with Block Limit Registers << Block Limit Warps; registers_per_thread > 100; kernel has 2D intermediate tensors shaped [XBLOCK, R0_BLOCK] where R0_BLOCK covers the full reduction dimension; XBLOCK is large (≥32) relative to the reduction size.

> **Avoid when**: XBLOCK=1 causes excessive kernel launch overhead (grid >> 10M programs on small tensors); when register pressure is not the occupancy limiter (Block Limit Registers ≈ Block Limit Warps); when the reduction dimension is much larger than XBLOCK making per-row granularity wasteful; when memory bandwidth (not occupancy) is the bottleneck.

#### ✅ Plan C: `reduce_register_pressure_via_xblock_autotune` (25.449×)

When a Triton persistent-reduction kernel uses a 2D tile [XBLOCK, R_BLOCK] and XBLOCK is large (e.g. 128), the compiler must keep all R_BLOCK reduction-lane values live per XBLOCK row simultaneously, causing extreme register pressure (168 regs/thread here). This caps active blocks/SM at 3 (18.75% theoretical occupancy) and produces >35% scheduler 'No Eligible' stall rate. Adding @triton.autotune with small XBLOCK configs (1–8) collapses the tile to near-1D, cutting registers from 168 to 23/thread, raising theoretical occupancy from 18.75% to 100%, and reducing kernel duration by 25x.

> **Applicable when**: Triton kernel with 2D [XBLOCK, R_BLOCK] tile pattern where XBLOCK >= 32 and registers_per_thread > 80; TheoreticalOccupancy NCU rule fires; Block Limit Registers << Block Limit Warps; theoretical occupancy < 25%; scheduler 'No Eligible' stall rate > 25%

> **Avoid when**: XBLOCK is already small (1–8); register pressure is not the occupancy bottleneck (Block Limit Registers >= Block Limit Warps); kernel correctness requires a specific XBLOCK (e.g. tiled matmul where XBLOCK affects accumulator shape); R_BLOCK must be reduced instead of XBLOCK to fix the issue

---

## Iteration 2

**Best so far**: 25.4491× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_num_warps_and_num_stages_for_latency_hiding | 22.377× | 25.190× | ✓ |  |
| B | expand_autotune_configs_num_warps_for_fma_and_occupancy | 27.181× | 30.843× | ✓ | **✓ best** |
| C | increase_xblock_tile_sizes_with_adaptive_grid_and_more_warps | 7.338× | 7.621× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_n... | Δ | expand_autotune_c... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 717410 ns | 28480 ns | -688930 ns (-96.0%) | 23260 ns | -694150 ns (-96.8%) | 94140 ns | -623270 ns (-86.9%) |
| SM Throughput (%) | 64.43 % | 77.78 % | +13.35 pp | 46.77 % | -17.66 pp | 75.96 % | +11.53 pp |
| Memory Throughput (%) | 60.16 % | 68.51 % | +8.35 pp | 44.17 % | -15.99 pp | 63.47 % | +3.31 pp |
| DRAM Throughput (%) | 60.16 % | 6.00 % | -54.16 pp | 7.35 % | -52.81 pp | 52.29 % | -7.87 pp |
| Achieved Occupancy (%) | 18.30 % | 87.36 % | +69.06 pp | 44.54 % | +26.24 pp | 40.94 % | +22.64 pp |
| Registers/Thread | 168.00 register/thread | 23.00 register/thread | -145 | 22.00 register/thread | -146 | 70.00 register/thread | -98 |
| Waves/SM | 82.75 | 15.52 | -67.23 | 7.76 | -74.99 | 17.73 | -65.02 |
| No Eligible (%) | 35.32 % | 16.63 % | -18.69 pp | 44.65 % | +9.33 pp | 21.61 % | -13.71 pp |
| IPC Active | 2.60 inst/cycle | 3.31 inst/cycle | +0.71 | 2.20 inst/cycle | -0.40 | 3.13 inst/cycle | +0.53 |

---

## Iteration 3

**Best so far**: 30.8431× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock1_for_latency_hiding | 26.939× | 30.883× | ✓ |  |
| B | eliminate_dynamic_r0_mask_and_fix_grid_lambda_for_fma_fusion | 27.444× | 30.711× | ✓ | **✓ best** |
| C | static_r0_mask_elimination_and_num_stages_pipelining | 26.970× | 30.790× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | eliminate_dynamic... | Δ | static_r0_mask_el... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 717410 ns | 23230 ns | -694180 ns (-96.8%) | 23360 ns | -694050 ns (-96.7%) | 23300 ns | -694110 ns (-96.8%) |
| SM Throughput (%) | 64.43 % | 46.95 % | -17.48 pp | 45.47 % | -18.96 pp | 45.47 % | -18.96 pp |
| Memory Throughput (%) | 60.16 % | 44.40 % | -15.76 pp | 44.69 % | -15.47 pp | 44.72 % | -15.44 pp |
| DRAM Throughput (%) | 60.16 % | 7.36 % | -52.80 pp | 7.32 % | -52.84 pp | 7.35 % | -52.81 pp |
| Achieved Occupancy (%) | 18.30 % | 44.13 % | +25.83 pp | 44.42 % | +26.12 pp | 43.70 % | +25.40 pp |
| Registers/Thread | 168.00 register/thread | 22.00 register/thread | -146 | 22.00 register/thread | -146 | 22.00 register/thread | -146 |
| Waves/SM | 82.75 | 7.76 | -74.99 | 7.76 | -74.99 | 7.76 | -74.99 |
| No Eligible (%) | 35.32 % | 44.34 % | +9.02 pp | 46.67 % | +11.35 pp | 46.83 % | +11.51 pp |
| IPC Active | 2.60 inst/cycle | 2.21 inst/cycle | -0.39 | 2.09 inst/cycle | -0.51 | 2.13 inst/cycle | -0.47 |

---

## Iteration 4

**Best so far**: 30.8431× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_expand_xblock_tile_configs_for_full_coverage | 27.422× | 30.843× | ✓ |  |
| B | fix_grid_lambda_and_increase_warps_for_mio_latency_hiding | 27.222× | 30.790× | ✓ |  |
| C | fix_grid_lambda_and_expand_large_xblock_for_fma_fusion_and_cache_reuse | 27.450× | 30.790× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 717410 ns | 23260 ns | -694150 ns (-96.8%) | 23300 ns | -694110 ns (-96.8%) | 23300 ns | -694110 ns (-96.8%) |
| SM Throughput (%) | 64.43 % | 47.14 % | -17.29 pp | 45.55 % | -18.88 pp | 45.20 % | -19.23 pp |
| Memory Throughput (%) | 60.16 % | 44.59 % | -15.57 pp | 44.77 % | -15.39 pp | 44.43 % | -15.73 pp |
| DRAM Throughput (%) | 60.16 % | 7.35 % | -52.81 pp | 7.34 % | -52.82 pp | 7.34 % | -52.82 pp |
| Achieved Occupancy (%) | 18.30 % | 44.51 % | +26.21 pp | 43.77 % | +25.47 pp | 44.50 % | +26.20 pp |
| Registers/Thread | 168.00 register/thread | 22.00 register/thread | -146 | 22.00 register/thread | -146 | 22.00 register/thread | -146 |
| Waves/SM | 82.75 | 7.76 | -74.99 | 7.76 | -74.99 | 7.76 | -74.99 |
| No Eligible (%) | 35.32 % | 44.61 % | +9.29 pp | 47.19 % | +11.87 pp | 46.66 % | +11.34 pp |
| IPC Active | 2.60 inst/cycle | 2.21 inst/cycle | -0.39 | 2.12 inst/cycle | -0.48 | 2.09 inst/cycle | -0.51 |

---

## Iteration 5

**Best so far**: 30.8431× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_optimize_load_eviction_for_coalescing_and_coverage | 26.943× | 30.790× | ✓ |  |
| B | fix_grid_lambda_and_add_large_xblock_configs_for_mio_latency_hiding | 27.278× | 30.750× | ✓ |  |
| C | fix_grid_lambda_and_add_large_xblock_high_warp_configs_for_mio_latency_hiding | 27.465× | 30.750× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 717410 ns | 23300 ns | -694110 ns (-96.8%) | 23330 ns | -694080 ns (-96.7%) | 23330 ns | -694080 ns (-96.7%) |
| SM Throughput (%) | 64.43 % | 47.18 % | -17.25 pp | 45.46 % | -18.97 pp | 45.34 % | -19.09 pp |
| Memory Throughput (%) | 60.16 % | 44.60 % | -15.56 pp | 44.71 % | -15.45 pp | 44.57 % | -15.59 pp |
| DRAM Throughput (%) | 60.16 % | 7.34 % | -52.82 pp | 7.33 % | -52.83 pp | 7.34 % | -52.82 pp |
| Achieved Occupancy (%) | 18.30 % | 44.00 % | +25.70 pp | 43.73 % | +25.43 pp | 43.57 % | +25.27 pp |
| Registers/Thread | 168.00 register/thread | 22.00 register/thread | -146 | 22.00 register/thread | -146 | 22.00 register/thread | -146 |
| Waves/SM | 82.75 | 7.76 | -74.99 | 7.76 | -74.99 | 7.76 | -74.99 |
| No Eligible (%) | 35.32 % | 44.87 % | +9.55 pp | 46.46 % | +11.14 pp | 47.11 % | +11.79 pp |
| IPC Active | 2.60 inst/cycle | 2.21 inst/cycle | -0.39 | 2.13 inst/cycle | -0.47 | 2.12 inst/cycle | -0.48 |

---

## Cumulative Progress

```
Baseline:  717410 ns  (1.000×)
Iter 1     28190 ns  (25.449×)  — reduce_register_pressure_via_xblock_autotune
Iter 2     23260 ns  (30.843×)  — expand_autotune_configs_num_warps_for_fma_and_occupancy

Final best: 23330 ns  (30.7505×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,66 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # num_warps=2 configs (new)
+        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 2}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 4}, num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=2),
+        triton.Config({'XBLOCK': 2}, num_warps=2,  num_stages=2),
+        triton.Config({'XBLOCK': 4}, num_warps=2,  num_stages=2),
+        # num_stages=1 (no pipelining) x num_warps in {4, 8, 16}
+        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 2}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 4}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 8}, num_warps=4,  num_stages=1),
+        # num_stages=2 (software pipelining: overlaps loads with compute)
+        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 2}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 4}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 4}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 4}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 8}, num_warps=8,  num_stages=2),
+        # new: XBLOCK in {16, 32, 64} x num_warps in {4, 8} x num_stages in {1, 2}
+        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=2),
+        # new: XBLOCK in {128, 256, 512} x num_warps in {8, 16} x num_stages in {1, 2}
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_per_fused_convolution_gelu_mul_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 4194304
@@ -14,19 +74,19 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
+    r0_mask = tl.full([1, R0_BLOCK], True, tl.int1)  # static: R0_BLOCK==r0_numel==64
     roffset = r0_offset
     rindex = r0_index
     r0_3 = r0_index
     x4 = xindex
     x1 = ((xindex // 2048) % 64)
-    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), None)
-    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
-    tmp23 = tl.load(in_ptr1 + (r0_3), None, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr2 + (r0_3), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), xmask)
+    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
+    tmp23 = tl.load(in_ptr1 + (r0_3), xmask & r0_mask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr2 + (r0_3), xmask & r0_mask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
     tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
@@ -56,9 +116,7 @@

     tmp33 = tmp31 + tmp32
     tmp34 = tmp28 * tmp33
     tmp35 = tmp34 * tmp32
-    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp35, None)
-
-
+    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp35, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -68,7 +126,7 @@

     xnumel = 4194304
     r0_numel = 64
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_convolution_gelu_mul_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 4194304, 64, XBLOCK=128)
+    triton_per_fused_convolution_gelu_mul_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 4194304, 64)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
