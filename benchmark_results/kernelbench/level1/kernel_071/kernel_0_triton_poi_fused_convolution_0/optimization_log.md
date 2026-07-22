# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 3390 ns (**1.7640×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_with_fixed_lambda_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_071/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.26 % | **Memory**: 41.70 % | **DRAM**: 20.73 %
- **Achieved Occupancy**: 55.35 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 70.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_tile_reconfig | 1.234× | 1.344× | ✓ |  |
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.269× | 0.593× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.034× | 1.056× | ✓ |  |
| D | add_autotune_tile_sizes_for_underutilized_pointwise_kernel | 1.395× | 1.886× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | add_autotune_tile... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4450 ns | -1530 ns (-25.6%) | 10080 ns | +4100 ns (+68.6%) | 5660 ns | -320 ns (-5.4%) | 3170 ns | -2810 ns (-47.0%) |
| SM Throughput (%) | 16.26 % | 21.17 % | +4.91 pp | 16.01 % | -0.25 pp | 18.23 % | +1.97 pp | 5.19 % | -11.07 pp |
| Memory Throughput (%) | 41.70 % | 31.34 % | -10.36 pp | 37.14 % | -4.56 pp | 37.78 % | -3.92 pp | 11.41 % | -30.29 pp |
| DRAM Throughput (%) | 20.73 % | 14.80 % | -5.93 pp | 6.11 % | -14.62 pp | 21.72 % | +0.99 pp | 2.62 % | -18.11 pp |
| Achieved Occupancy (%) | 55.35 % | 47.06 % | -8.29 pp | 52.62 % | -2.73 pp | 52.80 % | -2.55 pp | 25.69 % | -29.66 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 32.00 register/thread | +6 | 32.00 register/thread | +6 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 70.66 % | 55.46 % | -15.20 pp | 74.01 % | +3.35 pp | 66.10 % | -4.56 pp | 83.75 % | +13.09 pp |
| IPC Active | 1.12 inst/cycle | 1.70 inst/cycle | +0.58 | 1.01 inst/cycle | -0.11 | 1.30 inst/cycle | +0.18 | 0.61 inst/cycle | -0.51 |


### Key Learnings

#### ✅ Plan D: `add_autotune_tile_sizes_for_underutilized_pointwise_kernel` (1.905×)

When a Triton layout-conversion kernel has been burdened with a tl.trans() shared-memory transpose (inflating registers/thread from 16 to 28, adding dynamic SMEM, and worsening uncoalesced sectors from 25K to 200K), reverting to a single-phase direct load+store and expanding the autotune config space to include YBLOCK=1 entries recovers the optimal tile configuration. The SMEM transpose approach nearly doubled elapsed cycles (4709→8067) in a prior iteration; stripping it and restoring the direct path reduced executed instructions from 134K to 119K and improved achieved occupancy from 24.4% to 26.6%, yielding a 1.90x wall-clock speedup. The key insight is that for small-ynumel (≤30) layout-conversion kernels with a stride-N store pattern, the SMEM transpose does not resolve the fundamental store uncoalescing and only adds overhead — the autotune-selected YBLOCK=1 config with direct load+store is the correct optimum.

> **Applicable when**: Triton 2D pointwise or layout-conversion kernel where: (1) a tl.trans() SMEM transpose was previously added to try to coalesce strided stores; (2) registers/thread and/or dynamic SMEM increased after adding the transpose; (3) uncoalesced excess sectors did not decrease (or worsened) after the transpose; (4) ynumel is small (≤30) relative to YBLOCK; (5) the autotune config space does not include YBLOCK=1 entries.

> **Avoid when**: The SMEM transpose genuinely reduced uncoalesced excess sectors and improved elapsed cycles (i.e., the store stride pattern was actually resolved by transposition); the kernel is compute-bound rather than memory-latency-bound; ynumel is large enough that YBLOCK=1 would create too many blocks and overwhelm the scheduler.

---

## Iteration 2

**Best so far**: 1.8864× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_fixed_lambda_grid | 1.438× | 1.764× | ✓ | **✓ best** |
| B | store_eviction_policy_and_num_stages_pipelining | 1.367× | 1.375× | ✓ |  |
| C | vectorized_loads_with_autotune_for_strided_layout_conversion | 1.342× | 1.869× | ✓ |  |
| D | vectorized_loads_fix_autotune_static_grid_override | 1.219× | 0.609× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | store_eviction_po... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3390 ns | -2590 ns (-43.3%) | 4350 ns | -1630 ns (-27.3%) | 3200 ns | -2780 ns (-46.5%) | 9820 ns | +3840 ns (+64.2%) |
| SM Throughput (%) | 16.26 % | 5.18 % | -11.08 pp | 20.99 % | +4.73 pp | 5.18 % | -11.08 pp | 14.15 % | -2.11 pp |
| Memory Throughput (%) | 41.70 % | 10.57 % | -31.13 pp | 32.31 % | -9.39 pp | 11.34 % | -30.36 pp | 32.82 % | -8.88 pp |
| DRAM Throughput (%) | 20.73 % | 2.46 % | -18.27 pp | 15.15 % | -5.58 pp | 2.63 % | -18.10 pp | 6.25 % | -14.48 pp |
| Achieved Occupancy (%) | 55.35 % | 25.63 % | -29.72 pp | 47.43 % | -7.92 pp | 25.74 % | -29.61 pp | 52.36 % | -2.99 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 27.00 register/thread | +1 | 16.00 register/thread | -10 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.66 % | 79.51 % | +8.85 pp | 54.23 % | -16.43 pp | 80.00 % | +9.34 pp | 74.25 % | +3.59 pp |
| IPC Active | 1.12 inst/cycle | 0.66 inst/cycle | -0.46 | 1.77 inst/cycle | +0.65 | 0.65 inst/cycle | -0.47 | 1.00 inst/cycle | -0.12 |

---

## Iteration 3

**Best so far**: 1.8864× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_store_to_bypass_l1_for_write_once_output | 1.405× | 1.851× | ✓ |  |
| B | evict_first_store_to_reduce_l1tex_pressure_on_strided_stores | 1.378× | 1.869× | ✓ |  |
| C | shared_memory_transpose_to_coalesce_stores | 1.364× | 1.099× | ✓ |  |
| D | fix_autotune_grid_lambda_and_evict_first_store | 1.218× | 0.597× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_store... | Δ | evict_first_store... | Δ | shared_memory_tra... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3230 ns | -2750 ns (-46.0%) | 3200 ns | -2780 ns (-46.5%) | 5440 ns | -540 ns (-9.0%) | 10020 ns | +4040 ns (+67.6%) |
| SM Throughput (%) | 16.26 % | 5.76 % | -10.50 pp | 5.38 % | -10.88 pp | 17.25 % | +0.99 pp | 14.33 % | -1.93 pp |
| Memory Throughput (%) | 41.70 % | 11.02 % | -30.68 pp | 11.34 % | -30.36 pp | 34.77 % | -6.93 pp | 32.57 % | -9.13 pp |
| DRAM Throughput (%) | 20.73 % | 2.57 % | -18.16 pp | 2.61 % | -18.12 pp | 12.10 % | -8.63 pp | 6.16 % | -14.57 pp |
| Achieved Occupancy (%) | 55.35 % | 25.64 % | -29.71 pp | 24.35 % | -31.00 pp | 45.56 % | -9.79 pp | 52.52 % | -2.83 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 28.00 register/thread | +2 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.66 % | 79.47 % | +8.81 pp | 78.58 % | +7.92 pp | 67.39 % | -3.27 pp | 73.35 % | +2.69 pp |
| IPC Active | 1.12 inst/cycle | 0.74 inst/cycle | -0.38 | 0.69 inst/cycle | -0.43 | 1.37 inst/cycle | +0.25 | 1.05 inst/cycle | -0.07 |


### Key Learnings

#### ✅ Plan C: `shared_memory_transpose_to_coalesce_stores` (1.099×)

When a Triton layout-conversion kernel has a stride-N (N>1) store pattern in the fast thread dimension, staging data through a shared-memory tile and transposing with tl.trans() before the store can improve occupancy and eligible warps per scheduler. However, this approach increases elapsed cycles (4709→8067) and total uncoalesced sectors (25088→200704) because the SMEM transpose introduces additional instruction overhead and the larger block size (64→128 threads) with more registers per thread (16→28) reduces the number of blocks that can co-reside per SM, while the fundamental stride-3 store pattern in the output address remains unresolved — the y-dimension fast-dim store still has stride-1 only within the 3-element y-group, not across the full output. The 1.10x wall-clock speedup is modest and likely reflects the doubled waves/SM (0.37→0.74) from the larger tile covering more work per block, not from improved store coalescing.

> **Applicable when**: Triton layout-conversion kernel with UncoalescedGlobalAccess on stores; store index has stride-N (N>1) in the fast thread dimension; the slow dimension has stride-1 in the output; SMEM budget is sufficient for XBLOCK*YBLOCK*dtype_bytes per block; the ynumel dimension is small (e.g., 30) so making y the fast dim covers the full low-stride extent within a warp.

> **Avoid when**: The stride-N pattern in the output cannot be resolved by transposing axes (e.g., both dimensions have non-unit stride); the SMEM tile size exceeds per-block SMEM limits; the kernel is already achieving near-peak memory bandwidth; the additional instruction overhead from tl.trans() and dual-phase indexing outweighs the coalescing benefit (as seen here where elapsed cycles nearly doubled despite modest wall-clock improvement); when the ynumel fast-dim elements still have non-unit stride in the output after transposition.

---

## Iteration 4

**Best so far**: 1.8864× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_override_to_enable_autotune_selection | 1.366× | 1.886× | ✓ |  |
| B | fix_static_grid_to_lambda_for_autotune_effectiveness | 1.425× | 1.869× | ✓ |  |
| C | strip_smem_transpose_add_yblock1_autotune_configs | 1.386× | 1.905× | ✓ |  |
| D | fix_autotune_lambda_grid_and_add_yblock1_configs_for_workload_balance | 1.228× | 0.592× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_static_grid_t... | Δ | strip_smem_transp... | Δ | fix_autotune_lamb... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3170 ns | -2810 ns (-47.0%) | 3200 ns | -2780 ns (-46.5%) | 3140 ns | -2840 ns (-47.5%) | 10110 ns | +4130 ns (+69.1%) |
| SM Throughput (%) | 16.26 % | 5.65 % | -10.61 pp | 5.64 % | -10.62 pp | 5.01 % | -11.25 pp | 16.59 % | +0.33 pp |
| Memory Throughput (%) | 41.70 % | 11.10 % | -30.60 pp | 11.27 % | -30.43 pp | 11.40 % | -30.30 pp | 37.58 % | -4.12 pp |
| DRAM Throughput (%) | 20.73 % | 2.61 % | -18.12 pp | 2.61 % | -18.12 pp | 2.65 % | -18.08 pp | 6.09 % | -14.64 pp |
| Achieved Occupancy (%) | 55.35 % | 26.07 % | -29.28 pp | 25.79 % | -29.56 pp | 26.56 % | -28.79 pp | 52.53 % | -2.82 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.66 % | 82.09 % | +11.43 pp | 78.22 % | +7.56 pp | 80.09 % | +9.43 pp | 73.06 % | +2.40 pp |
| IPC Active | 1.12 inst/cycle | 0.75 inst/cycle | -0.37 | 0.72 inst/cycle | -0.40 | 0.71 inst/cycle | -0.41 | 1.05 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     3170 ns  (1.886×)  — add_autotune_tile_sizes_for_underutilized_pointwise_kernel

Final best: 3390 ns  (1.7640×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +34,16 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex_1d = xoffset + tl.arange(0, XBLOCK)
+    xindex_1d = tl.max_contiguous(tl.multiple_of(xindex_1d, XBLOCK), XBLOCK)
+    xindex = xindex_1d[None, :]
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
@@ -30,7 +51,7 @@

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
