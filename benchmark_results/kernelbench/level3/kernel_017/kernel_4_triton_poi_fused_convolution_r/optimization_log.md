# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2590 ns — **Best latency**: 2270 ns (**1.1410×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`right_size_tile_and_vectorize_loads_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_4_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (48,1,1), Waves/SM: 0.02
- **Registers/thread**: 30
- **Duration**: 2590 ns
- **SM Throughput**: 1.01 % | **Memory**: 1.10 % | **DRAM**: 0.15 %
- **Achieved Occupancy**: 6.18 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 88.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reshape_tile_config_and_add_autotune_for_small_grid_utilization | 0.992× | 1.079× | ✓ |  |
| B | fix_tile_dimensions_to_increase_grid_size | 1.005× | 1.053× | ✓ |  |
| C | fix_tile_sizes_to_eliminate_underutilization | 1.018× | 1.066× | ✓ |  |
| D | right_size_tile_and_vectorize_loads_stores | 1.076× | 1.141× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reshape_tile_conf... | Δ | fix_tile_dimensio... | Δ | fix_tile_sizes_to... | Δ | right_size_tile_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2400 ns | -190 ns (-7.3%) | 2460 ns | -130 ns (-5.0%) | 2430 ns | -160 ns (-6.2%) | 2270 ns | -320 ns (-12.4%) |
| SM Throughput (%) | 1.01 % | 0.79 % | -0.22 pp | 0.88 % | -0.13 pp | 0.08 % | -0.93 pp | 0.19 % | -0.82 pp |
| Memory Throughput (%) | 1.10 % | 0.61 % | -0.49 pp | 0.96 % | -0.14 pp | 0.45 % | -0.65 pp | 0.48 % | -0.62 pp |
| DRAM Throughput (%) | 0.15 % | 0.16 % | +0.01 pp | 0.15 % | +0.00 pp | 0.14 % | -0.01 pp | 0.15 % | +0.00 pp |
| Achieved Occupancy (%) | 6.18 % | 3.12 % | -3.06 pp | 6.18 % | +0.00 pp | 1.56 % | -4.62 pp | 3.09 % | -3.09 pp |
| Registers/Thread | 30.00 register/thread | 25.00 register/thread | -5 | 30.00 register/thread | +0 | 16.00 register/thread | -14 | 16.00 register/thread | -14 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.02 | +0.00 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 88.45 % | 80.07 % | -8.38 pp | 89.30 % | +0.85 pp | 93.32 % | +4.87 pp | 88.24 % | -0.21 pp |
| IPC Active | 0.35 inst/cycle | 0.39 inst/cycle | +0.04 | 0.36 inst/cycle | +0.01 | 0.06 inst/cycle | -0.29 | 0.10 inst/cycle | -0.25 |


### Key Learnings

#### ✅ Plan B: `fix_tile_dimensions_to_increase_grid_size` (1.053×)

When a Triton kernel's XBLOCK is much larger than xnumel (e.g., XBLOCK=128 for xnumel=9), the vast majority of threads per block are masked out, collapsing the grid to far fewer blocks than available SMs. Reducing XBLOCK to match the actual problem dimension (e.g., XBLOCK=1) and compensating by increasing YBLOCK redistributes work across more blocks, improving SM coverage. The NCU LaunchConfiguration rule and low waves_per_sm (<0.1) are the key signals; however, in this case the grid size and occupancy metrics did not change in the profiling data, suggesting the actual code diff only removed a blank line and the speedup is marginal noise rather than a structural improvement.

> **Applicable when**: LaunchConfiguration NCU rule is triggered; grid_size < sm_count; waves_per_sm < 0.1; XBLOCK (or YBLOCK) is much larger than the corresponding problem dimension (xnumel or ynumel), causing most threads to be masked out; achieved occupancy is very low (<15%)

> **Avoid when**: The problem dimension is large enough to fill the tile (xnumel >= XBLOCK); the kernel is already achieving high occupancy (>50%); reducing block size would increase kernel launch overhead beyond the parallelism benefit; the kernel has shared memory usage that benefits from larger tiles

---

## Iteration 2

**Best so far**: 1.141× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_and_flatten_to_1d_persistent_sweep | 1.025× | 1.156× | ✓ |  |
| E | add_triton_autotune | 1.061× | 1.036× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2240 ns | -350 ns (-13.5%) | 2500 ns | -90 ns (-3.5%) |
| SM Throughput (%) | 1.01 % | 0.06 % | -0.95 pp | 0.19 % | -0.82 pp |
| Memory Throughput (%) | 1.10 % | 0.53 % | -0.57 pp | 0.48 % | -0.62 pp |
| DRAM Throughput (%) | 0.15 % | 0.16 % | +0.01 pp | 0.14 % | -0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 3.07 % | -3.11 pp |
| Registers/Thread | 30.00 register/thread | 16.00 register/thread | -14 | 16.00 register/thread | -14 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 88.45 % | 95.55 % | +7.10 pp | 92.60 % | +4.15 pp |
| IPC Active | 0.35 inst/cycle | 0.05 inst/cycle | -0.30 | 0.19 inst/cycle | -0.16 |

---

## Iteration 3

**Best so far**: 1.141× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_hardcoded_grid_to_use_autotune_meta_config | 1.024× | 1.156× | ✓ |  |
| D | fix_broken_grid_lambda_to_activate_existing_autotune | 1.070× | 1.024× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_broken_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2240 ns | -350 ns (-13.5%) | 2530 ns | -60 ns (-2.3%) |
| SM Throughput (%) | 1.01 % | 0.08 % | -0.93 pp | 0.20 % | -0.81 pp |
| Memory Throughput (%) | 1.10 % | 0.53 % | -0.57 pp | 0.56 % | -0.54 pp |
| DRAM Throughput (%) | 0.15 % | 0.16 % | +0.01 pp | 0.14 % | -0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 3.00 % | -3.18 pp |
| Registers/Thread | 30.00 register/thread | 16.00 register/thread | -14 | 16.00 register/thread | -14 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 88.45 % | 93.02 % | +4.57 pp | 88.07 % | -0.38 pp |
| IPC Active | 0.35 inst/cycle | 0.06 inst/cycle | -0.29 | 0.19 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  2590 ns  (1.000×)
Iter 1     2270 ns  (1.141×)  — right_size_tile_and_vectorize_loads_stores

Final best: 2270 ns  (1.1410×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     # Simple 2D tiled kernel -- NO persistent loop
@@ -20,14 +39,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9 * yindex), mask=ymask & xmask, other=0.0)
     tl.store(out_ptr0 + (y0 + 6 * x2 + 54 * y1), tmp0, mask=ymask & xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
