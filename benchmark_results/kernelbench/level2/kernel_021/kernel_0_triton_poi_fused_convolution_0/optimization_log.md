# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6270 ns — **Best latency**: 3550 ns (**1.7662×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6270 ns
- **SM Throughput**: 16.58 % | **Memory**: 39.22 % | **DRAM**: 19.67 %
- **Achieved Occupancy**: 54.59 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 72.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.262× | 1.048× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.421× | 1.766× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_better_parallelism_and_coalescing | 1.372× | 1.591× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 5980 ns | -290 ns (-4.6%) | 3550 ns | -2720 ns (-43.4%) | 3940 ns | -2330 ns (-37.2%) |
| SM Throughput (%) | 16.58 % | 16.12 % | -0.46 pp | 16.51 % | -0.07 pp | 19.99 % | +3.41 pp |
| Memory Throughput (%) | 39.22 % | 42.19 % | +2.97 pp | 19.25 % | -19.97 pp | 23.04 % | -16.18 pp |
| DRAM Throughput (%) | 19.67 % | 20.61 % | +0.94 pp | 4.67 % | -15.00 pp | 4.22 % | -15.45 pp |
| Achieved Occupancy (%) | 54.59 % | 55.13 % | +0.54 pp | 43.91 % | -10.68 pp | 48.47 % | -6.12 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.23 % | 72.09 % | -0.14 pp | 54.63 % | -17.60 pp | 54.58 % | -17.65 pp |
| IPC Active | 1.15 inst/cycle | 1.15 inst/cycle | +0.00 | 1.69 inst/cycle | +0.54 | 1.85 inst/cycle | +0.70 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_occupancy_and_coalescing` (1.766×)

When a Triton kernel has low eligible warps per scheduler (<1.0) and high warp cycles per issued instruction (>25), adding a @triton.autotune decorator over XBLOCK, YBLOCK, and num_warps configurations allows the runtime to empirically select the tile sizes that best hide memory latency and reduce stall cycles. In this case, autotuning reduced warp cycles per issued instruction from 29.65 to 15.63 and increased eligible warps per scheduler from 0.9 to 2.07, cutting kernel duration by ~43%. The key mechanism is that smaller YBLOCK values reduce strided store scatter while varying XBLOCK controls the number of grid blocks and per-warp instruction count, together improving scheduler efficiency without changing the kernel body.

> **Applicable when**: Triton 2D-tiled pointwise or layout-transpose kernels where: eligible warps per scheduler < 1.5, warp cycles per issued instruction > 20, CPIStall rule triggered with L1TEX stall >40% of cycles, UncoalescedGlobalAccess rule triggered, and the kernel body uses strided stores (non-unit stride in innermost dimension). Grid size is small relative to SM count (waves_per_sm < 1.5).

> **Avoid when**: Kernel is already autotuned or the grid is large enough to saturate all SMs (waves_per_sm > 2). Kernel body has data-dependent control flow that makes tile size changes unsafe. YBLOCK reduction would break correctness (e.g., reductions across the Y dimension within a block). Autotune overhead is unacceptable in latency-critical single-call scenarios.

---

## Iteration 2

**Best so far**: 1.7662× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_launch_site_and_expand_configs_for_latency_hiding | 1.408× | 1.518× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_tile_exploration | 1.372× | 1.529× | ✓ |  |
| C | streaming_eviction_policy_with_yblock1_autotune_configs | 1.315× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_laun... | Δ | fix_autotune_grid... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 4130 ns | -2140 ns (-34.1%) | 4100 ns | -2170 ns (-34.6%) | 6210 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 16.58 % | 19.28 % | +2.70 pp | 19.95 % | +3.37 pp | 16.71 % | +0.13 pp |
| Memory Throughput (%) | 39.22 % | 22.32 % | -16.90 pp | 22.99 % | -16.23 pp | 40.00 % | +0.78 pp |
| DRAM Throughput (%) | 19.67 % | 4.04 % | -15.63 pp | 4.06 % | -15.61 pp | 19.82 % | +0.15 pp |
| Achieved Occupancy (%) | 54.59 % | 50.43 % | -4.16 pp | 48.58 % | -6.01 pp | 55.58 % | +0.99 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.23 % | 51.93 % | -20.30 pp | 52.30 % | -19.93 pp | 70.68 % | -1.55 pp |
| IPC Active | 1.15 inst/cycle | 1.71 inst/cycle | +0.56 | 1.83 inst/cycle | +0.68 | 1.13 inst/cycle | -0.02 |

---

## Iteration 3

**Best so far**: 1.7662× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_launch_site_and_add_num_stages_for_latency_hiding | 1.405× | 1.518× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints_and_fixed_autotune_grid | 1.388× | 1.556× | ✓ |  |
| C | fix_static_launch_grid_to_enable_autotune_tile_exploration | 1.345× | 1.010× | ✓ |  |
| D | add_triton_autotune | 1.409× | 1.544× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_laun... | Δ | vectorized_loads_... | Δ | fix_static_launch... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 4130 ns | -2140 ns (-34.1%) | 4030 ns | -2240 ns (-35.7%) | 6210 ns | -60 ns (-1.0%) | 4060 ns | -2210 ns (-35.2%) |
| SM Throughput (%) | 16.58 % | 19.78 % | +3.20 pp | 19.83 % | +3.25 pp | 16.47 % | -0.11 pp | 18.32 % | +1.74 pp |
| Memory Throughput (%) | 39.22 % | 22.87 % | -16.35 pp | 22.93 % | -16.29 pp | 40.40 % | +1.18 pp | 21.12 % | -18.10 pp |
| DRAM Throughput (%) | 19.67 % | 4.00 % | -15.67 pp | 4.11 % | -15.56 pp | 19.96 % | +0.29 pp | 4.09 % | -15.58 pp |
| Achieved Occupancy (%) | 54.59 % | 49.04 % | -5.55 pp | 48.84 % | -5.75 pp | 55.28 % | +0.69 pp | 49.87 % | -4.72 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 26.00 register/thread | +0 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 72.23 % | 52.28 % | -19.95 pp | 51.76 % | -20.47 pp | 70.91 % | -1.32 pp | 54.85 % | -17.38 pp |
| IPC Active | 1.15 inst/cycle | 1.85 inst/cycle | +0.70 | 1.86 inst/cycle | +0.71 | 1.16 inst/cycle | +0.01 | 1.74 inst/cycle | +0.59 |

---

## Cumulative Progress

```
Baseline:  6270 ns  (1.000×)
Iter 1     3550 ns  (1.766×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 3550 ns  (1.7662×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
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
