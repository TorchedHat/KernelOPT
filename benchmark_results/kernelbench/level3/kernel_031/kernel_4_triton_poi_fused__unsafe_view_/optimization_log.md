# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11010 ns — **Best latency**: 11680 ns (**0.9426×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_4_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 25
- **Duration**: 11010 ns
- **SM Throughput**: 15.06 % | **Memory**: 55.90 % | **DRAM**: 33.95 %
- **Achieved Occupancy**: 81.31 %
- **IPC Active**: 0.76 inst/cycle | **No Eligible**: 81.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_tile_reorientation | 0.995× | 0.974× | ✓ |  |
| B | autotune_tile_sizes_for_coalesced_transpose | 1.941× | 0.794× | ✓ | **✓ best** |
| C | memory_coalescing_swap_xblock_yblock | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11010 ns | 11300 ns | +290 ns (+2.6%) | 13860 ns | +2850 ns (+25.9%) | 11010 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 15.06 % | 14.90 % | -0.16 pp | 15.15 % | +0.09 pp | 15.21 % | +0.15 pp |
| Memory Throughput (%) | 55.90 % | 54.38 % | -1.52 pp | 52.65 % | -3.25 pp | 56.02 % | +0.12 pp |
| DRAM Throughput (%) | 33.95 % | 33.04 % | -0.91 pp | 52.65 % | +18.70 pp | 34.72 % | +0.77 pp |
| Achieved Occupancy (%) | 81.31 % | 81.05 % | -0.26 pp | 80.35 % | -0.96 pp | 81.10 % | -0.21 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 81.07 % | 80.74 % | -0.33 pp | 82.24 % | +1.17 pp | 80.85 % | -0.22 pp |
| IPC Active | 0.76 inst/cycle | 0.75 inst/cycle | -0.01 | 0.60 inst/cycle | -0.16 | 0.77 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan B: `autotune_tile_sizes_for_coalesced_transpose` (0.794×)

Adding @triton.autotune with varied XBLOCK/YBLOCK configs to a small transpose kernel caused a 26% regression (11.01 us → 13.86 us). The DRAM throughput increased (33.95% → 52.65%) while L2 throughput dropped (83.48% → 61.83%), indicating the autotune selected a config that bypassed L2 reuse and went directly to DRAM more often. The warp stall cycles per instruction barely changed (67.19 → 65.98), confirming the coalescing hypothesis was wrong — the bottleneck was not coalescing width but rather the autotune overhead and/or a selected config that degraded L2 hit rate. Additionally, the grid was not updated to use a lambda with meta['XBLOCK']/meta['YBLOCK'], so the autotuned kernel was launched with the old fixed grid (128, 8) regardless of the selected config, causing a correctness/efficiency mismatch.

> **Applicable when**: Never apply this exact pattern — the grid was not updated to a lambda, breaking the autotune contract.

> **Avoid when**: Do not add @triton.autotune to a kernel without simultaneously updating the grid to a lambda that reads meta['XBLOCK'] and meta['YBLOCK']. Without the lambda grid, the kernel launches with the old fixed tile dimensions regardless of which config autotune selects, wasting autotune overhead and potentially degrading L2 reuse. Also avoid autotuning very small kernels (ynumel=256, xnumel=16384, ~11 us) where autotune overhead dominates and the L2 already provides 67% hit rate — the existing config may already be near-optimal.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.014× | 0.967× | ✓ |  |
| B | vectorized_loads_xdim_contiguous_hints | 1.013× | 0.995× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.946× | 0.943× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11010 ns | 11390 ns | +380 ns (+3.5%) | 11070 ns | +60 ns (+0.5%) | 11680 ns | +670 ns (+6.1%) |
| SM Throughput (%) | 15.06 % | 15.15 % | +0.09 pp | 14.88 % | -0.18 pp | 15.46 % | +0.40 pp |
| Memory Throughput (%) | 55.90 % | 54.17 % | -1.73 pp | 62.63 % | +6.73 pp | 59.29 % | +3.39 pp |
| DRAM Throughput (%) | 33.95 % | 32.62 % | -1.33 pp | 34.00 % | +0.05 pp | 43.03 % | +9.08 pp |
| Achieved Occupancy (%) | 81.31 % | 81.01 % | -0.30 pp | 81.50 % | +0.19 pp | 79.64 % | -1.67 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 81.07 % | 81.18 % | +0.11 pp | 80.61 % | -0.46 pp | 83.54 % | +2.47 pp |
| IPC Active | 0.76 inst/cycle | 0.76 inst/cycle | +0.00 | 0.71 inst/cycle | -0.05 | 0.71 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_load_to_free_l2_for_stores | 0.875× | 0.997× | ✓ |  |
| B | evict_first_load_policy_and_mask_simplification | 0.855× | 0.988× | ✓ |  |
| C | shared_memory_tiled_transpose_to_eliminate_l1tex_stalls | 0.999× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | evict_first_load_... | Δ | shared_memory_til... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11010 ns | 11040 ns | +30 ns (+0.3%) | 11140 ns | +130 ns (+1.2%) | 11200 ns | +190 ns (+1.7%) |
| SM Throughput (%) | 15.06 % | 14.63 % | -0.43 pp | 14.90 % | -0.16 pp | 15.01 % | -0.05 pp |
| Memory Throughput (%) | 55.90 % | 55.59 % | -0.31 pp | 55.30 % | -0.60 pp | 54.69 % | -1.21 pp |
| DRAM Throughput (%) | 33.95 % | 33.90 % | -0.05 pp | 33.14 % | -0.81 pp | 33.55 % | -0.40 pp |
| Achieved Occupancy (%) | 81.31 % | 83.19 % | +1.88 pp | 83.49 % | +2.18 pp | 81.20 % | -0.11 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 24.00 register/thread | -1 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 81.07 % | 82.64 % | +1.57 pp | 80.35 % | -0.72 pp | 81.65 % | +0.58 pp |
| IPC Active | 0.76 inst/cycle | 0.74 inst/cycle | -0.02 | 0.75 inst/cycle | -0.01 | 0.74 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_scoreboard_stalls | 0.853× | 0.972× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency | 0.868× | 0.992× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_latency | 0.997× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11010 ns | 11330 ns | +320 ns (+2.9%) | 11100 ns | +90 ns (+0.8%) | 10980 ns | -30 ns (-0.3%) |
| SM Throughput (%) | 15.06 % | 14.73 % | -0.33 pp | 14.88 % | -0.18 pp | 14.50 % | -0.56 pp |
| Memory Throughput (%) | 55.90 % | 54.45 % | -1.45 pp | 55.47 % | -0.43 pp | 56.04 % | +0.14 pp |
| DRAM Throughput (%) | 33.95 % | 32.51 % | -1.44 pp | 33.33 % | -0.62 pp | 34.03 % | +0.08 pp |
| Achieved Occupancy (%) | 81.31 % | 83.19 % | +1.88 pp | 83.44 % | +2.13 pp | 81.19 % | -0.12 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 24.00 register/thread | -1 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 81.07 % | 81.12 % | +0.05 pp | 80.89 % | -0.18 pp | 81.11 % | +0.04 pp |
| IPC Active | 0.76 inst/cycle | 0.74 inst/cycle | -0.02 | 0.77 inst/cycle | +0.01 | 0.73 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  11010 ns  (1.000×)

Final best: 11680 ns  (0.9426×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,22 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 256
     xnumel = 16384
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yarange = tl.max_contiguous(tl.multiple_of(yoffset + tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
+    yindex = yarange[None, :]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[:, None]
     x1 = xindex
     y0 = yindex
-    tmp0 = tl.load(in_ptr0 + (4194304 + y0 + 256*x1), ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (4194304 + y0 + 256*x1), ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (x1 + 16384*y0), tmp0, ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
