# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3420 ns — **Best latency**: 2750 ns (**1.2436×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_persistent_grid_to_sm_count_with_cg_eviction_for_streaming_gather`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_7_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3420 ns
- **SM Throughput**: 8.91 % | **Memory**: 6.94 % | **DRAM**: 4.85 %
- **Achieved Occupancy**: 52.37 %
- **IPC Active**: 0.95 inst/cycle | **No Eligible**: 72.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_warps_for_underutilized_pointwise_permute | 1.033× | 1.103× | ✓ | **✓ best** |
| B | autotune_xblock_to_reduce_grid_and_improve_utilization | 1.015× | 1.125× | ✓ |  |
| C | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 1.031× | 0.980× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_underutilized_pointwise_permute | 0.970× | 1.125× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_t... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3420 ns | 3100 ns | -320 ns (-9.4%) | 3040 ns | -380 ns (-11.1%) | 3490 ns | +70 ns (+2.0%) | 3040 ns | -380 ns (-11.1%) |
| SM Throughput (%) | 8.91 % | 8.94 % | +0.03 pp | 8.81 % | -0.10 pp | 26.47 % | +17.56 pp | 9.06 % | +0.15 pp |
| Memory Throughput (%) | 6.94 % | 7.45 % | +0.51 pp | 7.38 % | +0.44 pp | 9.68 % | +2.74 pp | 7.39 % | +0.45 pp |
| DRAM Throughput (%) | 4.85 % | 5.39 % | +0.54 pp | 5.51 % | +0.66 pp | 4.76 % | -0.09 pp | 5.49 % | +0.64 pp |
| Achieved Occupancy (%) | 52.37 % | 46.20 % | -6.17 pp | 43.25 % | -9.12 pp | 69.97 % | +17.60 pp | 42.67 % | -9.70 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 | 0.75 | +0.00 |
| No Eligible (%) | 72.14 % | 70.60 % | -1.54 pp | 70.21 % | -1.93 pp | 42.56 % | -29.58 pp | 70.42 % | -1.72 pp |
| IPC Active | 0.95 inst/cycle | 1.00 inst/cycle | +0.05 | 1.04 inst/cycle | +0.09 | 2.21 inst/cycle | +1.26 | 1.04 inst/cycle | +0.09 |

---

## Iteration 2

**Best so far**: 1.1032× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_underutilized_permute_kernel | 1.017× | 1.244× | ✓ |  |
| B | persistent_tile_scheduling | 1.040× | 1.163× | ✓ | **✓ best** |
| C | persistent_tile_scheduling_for_underutilized_permute_kernel | 1.001× | 1.009× | ✓ |  |
| D | persistent_tile_scheduling_for_partial_wave_elimination | 1.011× | 1.103× | ✓ |  |
| E | add_triton_autotune | 0.989× | 1.036× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3420 ns | 2750 ns | -670 ns (-19.6%) | 2940 ns | -480 ns (-14.0%) | 3390 ns | -30 ns (-0.9%) | 3100 ns | -320 ns (-9.4%) | 3300 ns | -120 ns (-3.5%) |
| SM Throughput (%) | 8.91 % | 5.95 % | -2.96 pp | 8.49 % | -0.42 pp | 10.33 % | +1.42 pp | 14.08 % | +5.17 pp | 8.93 % | +0.02 pp |
| Memory Throughput (%) | 6.94 % | 8.19 % | +1.25 pp | 7.88 % | +0.94 pp | 6.99 % | +0.05 pp | 7.26 % | +0.32 pp | 7.19 % | +0.25 pp |
| DRAM Throughput (%) | 4.85 % | 6.05 % | +1.20 pp | 5.69 % | +0.84 pp | 4.89 % | +0.04 pp | 5.42 % | +0.57 pp | 5.04 % | +0.19 pp |
| Achieved Occupancy (%) | 52.37 % | 26.61 % | -25.76 pp | 42.08 % | -10.29 pp | 52.09 % | -0.28 pp | 45.06 % | -7.31 pp | 52.38 % | +0.01 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 22.00 register/thread | +6 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 | 0.75 | +0.00 |
| No Eligible (%) | 72.14 % | 87.35 % | +15.21 pp | 76.59 % | +4.45 pp | 68.00 % | -4.14 pp | 78.03 % | +5.89 pp | 72.59 % | +0.45 pp |
| IPC Active | 0.95 inst/cycle | 0.44 inst/cycle | -0.51 | 0.83 inst/cycle | -0.12 | 1.18 inst/cycle | +0.23 | 0.83 inst/cycle | -0.12 | 0.96 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan A: `persistent_tile_scheduling_for_underutilized_permute_kernel` (1.244×)

For small pointwise/permute kernels with sub-1-wave grid coverage (waves_per_sm < 1.0), replacing the single-tile-per-CTA pattern with a persistent while-loop (grid fixed at NUM_SM, each CTA iterates over ceil(num_tiles/NUM_SM) tiles) reduces total elapsed cycles and kernel duration even when occupancy drops. In this case, duration fell from 3.04 µs to 2.75 µs (1.24x speedup) and memory throughput rose from 266 to 294 GB/s, while executed instructions dropped from 211K to 82K — indicating the persistent loop eliminated redundant per-CTA overhead (prologue/epilogue, barrier synchronization, scheduler setup). Notably, achieved occupancy dropped further (43% → 27%) and no-eligible rate worsened (70% → 87%), yet the kernel was still faster because fewer total instructions were executed and memory bandwidth was better utilized per active CTA.

> **Applicable when**: Triton pointwise/permute kernel with: waves_per_sm < 1.0, grid_size significantly larger than sm_count (e.g. grid > 4x sm_count), kernel body is pure load/store with no shared memory or inter-CTA communication, xnumel is fixed/known at launch time, XBLOCK is tl.constexpr so tl.cdiv(xnumel, XBLOCK) is valid inside the kernel

> **Avoid when**: Kernel uses shared memory with inter-warp synchronization (barriers inside the tile body would deadlock across iterations); kernel already has waves_per_sm >= 2 (persistent loop adds while-loop overhead without benefit); num_tiles < NUM_SM at the winning XBLOCK (some SMs do zero work, wasting launch slots — though correctness is preserved); kernel has data-dependent early exit that cannot be expressed in a while-loop

#### ✅ Plan C: `persistent_tile_scheduling_for_underutilized_permute_kernel` (1.244×)

For small pointwise/permute kernels with sub-1-wave grid coverage (waves_per_sm < 1.0), replacing the single-tile-per-CTA pattern with a persistent while-loop (grid fixed at NUM_SM, each CTA iterates over ceil(num_tiles/NUM_SM) tiles) reduces total elapsed cycles and kernel duration even when occupancy drops. In this case, duration fell from 3.04 µs to 2.75 µs (1.24x speedup) and memory throughput rose from 266 to 294 GB/s, while executed instructions dropped from 211K to 82K — indicating the persistent loop eliminated redundant per-CTA overhead (prologue/epilogue, barrier synchronization, scheduler setup). Notably, achieved occupancy dropped further (43% → 27%) and no-eligible rate worsened (70% → 87%), yet the kernel was still faster because fewer total instructions were executed and memory bandwidth was better utilized per active CTA.

> **Applicable when**: Triton pointwise/permute kernel with: waves_per_sm < 1.0, grid_size significantly larger than sm_count (e.g. grid > 4x sm_count), kernel body is pure load/store with no shared memory or inter-CTA communication, xnumel is fixed/known at launch time, XBLOCK is tl.constexpr so tl.cdiv(xnumel, XBLOCK) is valid inside the kernel

> **Avoid when**: Kernel uses shared memory with inter-warp synchronization (barriers inside the tile body would deadlock across iterations); kernel already has waves_per_sm >= 2 (persistent loop adds while-loop overhead without benefit); num_tiles < NUM_SM at the winning XBLOCK (some SMs do zero work, wasting launch slots — though correctness is preserved); kernel has data-dependent early exit that cannot be expressed in a while-loop

---

## Iteration 3

**Best so far**: 1.1633× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_and_l1_cache_hints_for_l1tex_stall_reduction | 1.025× | 1.213× | ✓ |  |
| B | fix_persistent_kernel_grid_to_match_sm_count | 1.001× | 1.163× | ✓ |  |
| C | fix_grid_lambda_to_activate_autotune_xblock_scaling_with_pipeline_stages | 1.011× | 1.049× | ✓ |  |
| D | fix_persistent_grid_launch_with_streaming_eviction_policy | 1.000× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | fix_persistent_ke... | Δ | fix_grid_lambda_t... | Δ | fix_persistent_gr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3420 ns | 2820 ns | -600 ns (-17.5%) | 2940 ns | -480 ns (-14.0%) | 3260 ns | -160 ns (-4.7%) | 3390 ns | -30 ns (-0.9%) |
| SM Throughput (%) | 8.91 % | 5.84 % | -3.07 pp | 7.78 % | -1.13 pp | 26.34 % | +17.43 pp | 10.76 % | +1.85 pp |
| Memory Throughput (%) | 6.94 % | 8.03 % | +1.09 pp | 7.60 % | +0.66 pp | 9.61 % | +2.67 pp | 7.04 % | +0.10 pp |
| DRAM Throughput (%) | 4.85 % | 5.94 % | +1.09 pp | 5.64 % | +0.79 pp | 5.14 % | +0.29 pp | 4.93 % | +0.08 pp |
| Achieved Occupancy (%) | 52.37 % | 36.87 % | -15.50 pp | 42.71 % | -9.66 pp | 69.97 % | +17.60 pp | 52.85 % | +0.48 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 | 0.75 | +0.00 |
| No Eligible (%) | 72.14 % | 87.31 % | +15.17 pp | 77.17 % | +5.03 pp | 43.03 % | -29.11 pp | 69.24 % | -2.90 pp |
| IPC Active | 0.95 inst/cycle | 0.36 inst/cycle | -0.59 | 0.83 inst/cycle | -0.12 | 2.20 inst/cycle | +1.25 | 1.13 inst/cycle | +0.18 |

---

## Iteration 4

**Best so far**: 1.1633× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_for_persistent_kernel_efficiency | 1.044× | 1.244× | ✓ | **✓ best** |
| B | fix_grid_lambda_and_remove_broken_persistent_loop_for_correct_autotune | 1.034× | 1.114× | ✓ |  |
| C | fix_grid_lambda_to_use_autotuned_xblock_for_wave_elimination | 1.025× | 1.059× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3420 ns | 2750 ns | -670 ns (-19.6%) | 3070 ns | -350 ns (-10.2%) | 3230 ns | -190 ns (-5.6%) |
| SM Throughput (%) | 8.91 % | 5.81 % | -3.10 pp | 8.98 % | +0.07 pp | 26.31 % | +17.40 pp |
| Memory Throughput (%) | 6.94 % | 8.18 % | +1.24 pp | 7.38 % | +0.44 pp | 9.59 % | +2.65 pp |
| DRAM Throughput (%) | 4.85 % | 6.09 % | +1.24 pp | 5.42 % | +0.57 pp | 5.15 % | +0.30 pp |
| Achieved Occupancy (%) | 52.37 % | 25.72 % | -26.65 pp | 46.35 % | -6.02 pp | 69.82 % | +17.45 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 |
| No Eligible (%) | 72.14 % | 87.37 % | +15.23 pp | 71.70 % | -0.44 pp | 49.98 % | -22.16 pp |
| IPC Active | 0.95 inst/cycle | 0.44 inst/cycle | -0.51 | 1.00 inst/cycle | +0.05 | 2.21 inst/cycle | +1.26 |

---

## Iteration 5

**Best so far**: 1.2436× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_persistent_grid_to_sm_count_with_cg_eviction_for_streaming_gather | 1.021× | 1.244× | ✓ |  |
| B | fix_grid_lambda_to_use_meta_xblock_so_autotune_reduces_grid_proportionally | 0.996× | 1.103× | ✓ |  |
| C | fix_grid_lambda_to_use_meta_xblock_for_wave_elimination_v2 | 0.988× | 1.059× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_persistent_gr... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3420 ns | 2750 ns | -670 ns (-19.6%) | 3100 ns | -320 ns (-9.4%) | 3230 ns | -190 ns (-5.6%) |
| SM Throughput (%) | 8.91 % | 5.96 % | -2.95 pp | 9.15 % | +0.24 pp | 25.23 % | +16.32 pp |
| Memory Throughput (%) | 6.94 % | 8.15 % | +1.21 pp | 7.21 % | +0.27 pp | 9.21 % | +2.27 pp |
| DRAM Throughput (%) | 4.85 % | 6.06 % | +1.21 pp | 5.36 % | +0.51 pp | 5.14 % | +0.29 pp |
| Achieved Occupancy (%) | 52.37 % | 25.87 % | -26.50 pp | 46.68 % | -5.69 pp | 70.07 % | +17.70 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 |
| No Eligible (%) | 72.14 % | 87.52 % | +15.38 pp | 71.64 % | -0.50 pp | 42.99 % | -29.15 pp |
| IPC Active | 0.95 inst/cycle | 0.45 inst/cycle | -0.50 | 1.02 inst/cycle | +0.07 | 2.20 inst/cycle | +1.25 |

---

## Cumulative Progress

```
Baseline:  3420 ns  (1.000×)
Iter 1     3100 ns  (1.103×)  — autotune_xblock_and_warps_for_underutilized_pointwise_permute
Iter 2     2940 ns  (1.163×)  — persistent_tile_scheduling
Iter 4     2750 ns  (1.244×)  — fix_grid_lambda_to_use_meta_xblock_for_persistent_kernel_efficiency

Final best: 2750 ns  (1.2436×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,27 +5,52 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 4096}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_permute_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 201728
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x0 = (xindex % 512)
-    x1 = ((xindex // 512) % 197)
-    x2 = xindex // 100864
-    x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1024*x1), xmask)
-    tl.store(out_ptr0 + (x3), tmp0, xmask)
-
-
+    pid = tl.program_id(0)
+    num_sm = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x0 = (xindex % 512)
+        x1 = ((xindex // 512) % 197)
+        x2 = xindex // 100864
+        x3 = xindex
+        tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1024*x1), xmask,
+                       cache_modifier='.ca')
+        tl.store(out_ptr0 + (x3), tmp0, xmask)
+        tile_id += num_sm
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 201728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_permute_7[grid](arg_0, arg_1, 201728, XBLOCK=128)
+    triton_poi_fused_clone_permute_7[grid](arg_0, arg_1, 201728)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
