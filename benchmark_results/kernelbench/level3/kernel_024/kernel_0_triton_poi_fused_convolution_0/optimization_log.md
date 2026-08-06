# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 5180 ns (**1.1486×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_with_wave_aligned_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.40 % | **Memory**: 41.10 % | **DRAM**: 20.81 %
- **Achieved Occupancy**: 54.79 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.278× | 1.127× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.299× | 1.377× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_swap | 1.335× | 1.022× | ✓ |  |
| D | add_autotune_increase_tile_sizes_for_underutilized_kernel | 1.417× | 1.720× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 5280 ns | -670 ns (-11.3%) | 4320 ns | -1630 ns (-27.4%) | 5820 ns | -130 ns (-2.2%) | 3460 ns | -2490 ns (-41.8%) |
| SM Throughput (%) | 16.40 % | 36.31 % | +19.91 pp | 18.91 % | +2.51 pp | 16.15 % | -0.25 pp | 9.76 % | -6.64 pp |
| Memory Throughput (%) | 41.10 % | 37.03 % | -4.07 pp | 21.77 % | -19.33 pp | 42.89 % | +1.79 pp | 20.34 % | -20.76 pp |
| DRAM Throughput (%) | 20.81 % | 12.47 % | -8.34 pp | 3.83 % | -16.98 pp | 21.20 % | +0.39 pp | 4.82 % | -15.99 pp |
| Achieved Occupancy (%) | 54.79 % | 74.97 % | +20.18 pp | 48.85 % | -5.94 pp | 54.96 % | +0.17 pp | 51.03 % | -3.76 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 25.00 register/thread | -1 | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.57 % | 39.74 % | -30.83 pp | 51.74 % | -18.83 pp | 72.28 % | +1.71 pp | 69.75 % | -0.82 pp |
| IPC Active | 1.15 inst/cycle | 2.43 inst/cycle | +1.28 | 1.85 inst/cycle | +0.70 | 1.14 inst/cycle | -0.01 | 1.03 inst/cycle | -0.12 |

---

## Iteration 2

**Best so far**: 1.7197× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguity_hints | 1.407× | 1.825× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints | 1.295× | 1.397× | ✓ |  |
| C | fix_autotune_grid_lambda_and_expand_configs_for_partial_wave | 1.444× | 1.127× | ✓ | **✓ best** |
| D | vectorized_loads_with_contiguous_hints_and_pipelined_stages | 1.325× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 3260 ns | -2690 ns (-45.2%) | 4260 ns | -1690 ns (-28.4%) | 5280 ns | -670 ns (-11.3%) | 5890 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 16.40 % | 9.76 % | -6.64 pp | 19.13 % | +2.73 pp | 35.92 % | +19.52 pp | 16.29 % | -0.11 pp |
| Memory Throughput (%) | 41.10 % | 21.26 % | -19.84 pp | 22.10 % | -19.00 pp | 36.76 % | -4.34 pp | 42.57 % | +1.47 pp |
| DRAM Throughput (%) | 20.81 % | 5.03 % | -15.78 pp | 3.90 % | -16.91 pp | 12.46 % | -8.35 pp | 20.97 % | +0.16 pp |
| Achieved Occupancy (%) | 54.79 % | 53.55 % | -1.24 pp | 48.81 % | -5.98 pp | 74.41 % | +19.62 pp | 55.48 % | +0.69 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.57 % | 69.55 % | -1.02 pp | 51.56 % | -19.01 pp | 38.52 % | -32.05 pp | 72.04 % | +1.47 pp |
| IPC Active | 1.15 inst/cycle | 1.04 inst/cycle | -0.11 | 1.84 inst/cycle | +0.69 | 2.43 inst/cycle | +1.28 | 1.12 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_contiguity_hints` (1.825×)

Wrapping `tl.arange(0, XBLOCK)` with `tl.max_contiguous(tl.multiple_of(..., XBLOCK), XBLOCK)` before using it as a load index tells the Triton compiler that the range is XBLOCK-aligned and contiguous, enabling it to emit 128-bit (LDG.128) vectorized loads instead of scalar 32-bit loads. In this case, the change produced a modest but real improvement: elapsed cycles dropped from 5113 to 4893 (~4.3%), memory throughput increased from 233 to 247 GB/s, and achieved occupancy improved slightly from 51% to 53.6%. However, the dominant bottleneck (50% excessive sectors from uncoalesced stores on a stride-3 output layout) was unaffected, limiting the observable speedup from the load-side improvement alone.

> **Applicable when**: Triton kernel has a load path with stride-1 (contiguous) access in the fast-varying thread dimension (xindex); UncoalescedGlobalAccess or MemoryCacheAccessPattern rules are triggered on the load side; CPIStall on L1TEX is a significant stall contributor; xoffset is provably XBLOCK-aligned (i.e., xoffset = program_id(0) * XBLOCK); the load bottleneck is not fully masked by an equally severe store-side bottleneck.

> **Avoid when**: The load access pattern is not stride-1 in the fast thread dimension (hints would be incorrect/misleading); the store path has equally severe or worse uncoalesced access that dominates kernel time (the load improvement will be masked); the kernel is compute-bound rather than memory-latency-bound; xoffset is not a multiple of XBLOCK (the multiple_of hint would be incorrect).

---

## Iteration 3

**Best so far**: 1.7197× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | store_coalescing_via_yblock_inner_tiling | 0.740× | 0.581× | ✓ |  |
| C | coalesced_store_via_eviction_policy_and_grid_reshape | 1.418× | 1.140× | ✓ |  |
| D | streaming_store_eviction_policy_to_reduce_l1_pollution | 1.346× | 1.044× | ✓ |  |


### Metric Comparison

| Metric | Baseline | store_coalescing_... | Δ | coalesced_store_v... | Δ | streaming_store_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 10240 ns | +4290 ns (+72.1%) | 5220 ns | -730 ns (-12.3%) | 5700 ns | -250 ns (-4.2%) |
| SM Throughput (%) | 16.40 % | 13.87 % | -2.53 pp | 36.97 % | +20.57 pp | 17.25 % | +0.85 pp |
| Memory Throughput (%) | 41.10 % | 32.18 % | -8.92 pp | 36.59 % | -4.51 pp | 41.74 % | +0.64 pp |
| DRAM Throughput (%) | 20.81 % | 6.03 % | -14.78 pp | 12.63 % | -8.18 pp | 21.57 % | +0.76 pp |
| Achieved Occupancy (%) | 54.79 % | 52.11 % | -2.68 pp | 75.03 % | +20.24 pp | 54.68 % | -0.11 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.57 % | 75.02 % | +4.45 pp | 36.74 % | -33.83 pp | 69.09 % | -1.48 pp |
| IPC Active | 1.15 inst/cycle | 1.02 inst/cycle | -0.13 | 2.45 inst/cycle | +1.30 | 1.20 inst/cycle | +0.05 |


### Key Learnings

#### ✅ Plan D: `streaming_store_eviction_policy_to_reduce_l1_pollution` (1.044×)

Adding eviction_policy='evict_first' to tl.store for write-once, strided output tensors is intended to bypass L1 caching (PTX .cs operator), freeing L1 bandwidth for contiguous input loads. However, in this case the change was confounded by a simultaneous autotune config space replacement that selected a higher-register, shared-memory-using config (16→26 regs/thread, 0→4KB dynamic smem), which increased elapsed cycles from 4893 to 8513 (+74%) and duration from 3.26 to 5.7 µs. The net wall-clock speedup of 1.04x reflects the autotune config regression dominating over any benefit from the eviction policy hint. The L1 hit rate paradoxically rose from 0% to 38% after adding evict_first, suggesting the new autotune config changed access patterns more than the eviction policy itself.

> **Applicable when**: Store access pattern is strided/uncoalesced (UncoalescedGlobalAccess triggered); output tensor is write-once with no subsequent reads in the same kernel; CPIStall on L1TEX is the dominant stall; L1 hit rate on loads is low despite contiguous access pattern (suggesting L1 pollution from strided stores); eviction_policy change is the ONLY change (autotune config space must remain stable).

> **Avoid when**: Autotune config space is being changed simultaneously — the config selection effect will dominate and mask the eviction policy benefit; the store access pattern is contiguous/coalesced (evict_first provides no benefit and may hurt reuse); the kernel reads the output tensor back in the same kernel (evict_first would cause cache misses on re-reads); register pressure is already high and the new autotune config space would select a higher-register config.

---

## Iteration 4

**Best so far**: 1.7197× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_eliminate_partial_wave | 1.450× | 1.074× | ✓ | **✓ best** |
| B | fix_grid_lambda_to_use_meta_and_expand_large_xblock_configs | 1.342× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 5540 ns | -410 ns (-6.9%) | 5860 ns | -90 ns (-1.5%) |
| SM Throughput (%) | 16.40 % | 37.09 % | +20.69 pp | 17.00 % | +0.60 pp |
| Memory Throughput (%) | 41.10 % | 36.74 % | -4.36 pp | 41.72 % | +0.62 pp |
| DRAM Throughput (%) | 20.81 % | 11.88 % | -8.93 pp | 21.09 % | +0.28 pp |
| Achieved Occupancy (%) | 54.79 % | 73.99 % | +19.20 pp | 56.32 % | +1.53 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.57 % | 36.53 % | -34.04 pp | 71.29 % | +0.72 pp |
| IPC Active | 1.15 inst/cycle | 2.46 inst/cycle | +1.31 | 1.17 inst/cycle | +0.02 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_eliminate_partial_wave` (1.149×)

The diff shows only cosmetic changes: inline comments (e.g., `# grid=(196,15)=2940, tail=1.2%`) were removed from three autotune config lines, with no structural code changes — no grid lambda fix, no new configs, no kernel body changes. The 1.15x speedup (5.54→5.18 µs, 8245→7701 elapsed cycles) is likely attributable to run-to-run variance or a different autotune config being selected after the cache was invalidated by the file change, rather than any causal effect of the comment removal. The LaunchConfiguration rule still fires at 50% estimated speedup, waves_per_sm remains 1.48, and grid_size is unchanged at 1568, confirming the partial-wave problem was not addressed. The memory throughput improvement (580→620 GB/s) is consistent with a slightly better autotune config selection rather than a structural fix.

> **Applicable when**: Never: comment removal from autotune configs is not a performance optimization strategy. The observed speedup is within noise and does not generalize.

> **Avoid when**: Always: removing comments from autotune config lines has no causal effect on GPU performance. Do not attribute speedup to this change. The LaunchConfiguration partial-wave problem (50% estimated speedup) remains unresolved and requires a grid lambda fix.

---

## Iteration 5

**Best so far**: 1.7197× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_with_wave_aligned_autotune_configs | 1.453× | 1.149× | ✓ | **✓ best** |
| B | shared_memory_transpose_for_coalesced_store | 1.440× | 1.106× | ✓ |  |
| C | fix_hardcoded_grid_to_lambda_with_yblock1_configs | 1.406× | 1.028× | ✓ |  |
| D | fix_grid_lambda_and_add_load_evict_last_for_l1_reuse | 1.343× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_w... | Δ | shared_memory_tra... | Δ | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 5180 ns | -770 ns (-12.9%) | 5380 ns | -570 ns (-9.6%) | 5790 ns | -160 ns (-2.7%) | 5950 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 16.40 % | 36.35 % | +19.95 pp | 36.81 % | +20.41 pp | 16.62 % | +0.22 pp | 17.35 % | +0.95 pp |
| Memory Throughput (%) | 41.10 % | 35.99 % | -5.11 pp | 36.38 % | -4.72 pp | 41.55 % | +0.45 pp | 40.34 % | -0.76 pp |
| DRAM Throughput (%) | 20.81 % | 12.73 % | -8.08 pp | 12.24 % | -8.57 pp | 21.23 % | +0.42 pp | 20.68 % | -0.13 pp |
| Achieved Occupancy (%) | 54.79 % | 74.18 % | +19.39 pp | 74.18 % | +19.39 pp | 55.35 % | +0.56 pp | 56.35 % | +1.56 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.57 % | 36.67 % | -33.90 pp | 38.64 % | -31.93 pp | 69.37 % | -1.20 pp | 69.24 % | -1.33 pp |
| IPC Active | 1.15 inst/cycle | 2.48 inst/cycle | +1.33 | 2.47 inst/cycle | +1.32 | 1.22 inst/cycle | +0.07 | 1.17 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     3460 ns  (1.720×)  — add_autotune_increase_tile_sizes_for_underutilized_kernel

Final best: 5180 ns  (1.1486×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1},  num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 2},  num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1},  num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8},  num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=8,  num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +40,16 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xrange = tl.arange(0, XBLOCK)
+    xrange = tl.max_contiguous(tl.multiple_of(xrange, XBLOCK), XBLOCK)
+    xindex = xoffset + xrange[None, :]
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
@@ -30,7 +57,7 @@

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
