# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_slice_unsqueeze_view_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2400 ns — **Best latency**: 2620 ns (**0.9160×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_slice_unsqueeze_view_14`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_14_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 24
- **Duration**: 2400 ns
- **SM Throughput**: 0.17 % | **Memory**: 0.50 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.19 %
- **IPC Active**: 0.48 inst/cycle | **No Eligible**: 85.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_correct_3d_indexing | — | 0.916× | ✓ | **✓ best** |
| B | fix_3d_grid_launch_to_cover_full_y_dimension | — | 1.000× | ✓ |  |
| C | fix_grid_to_match_existing_3d_program_id_scheme | — | 0.976× | ✓ |  |
| D | fix_grid_to_cover_y_dimension_with_vectorized_stores | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ | fix_3d_grid_launc... | Δ | fix_grid_to_match... | Δ | fix_grid_to_cover... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2620 ns | +220 ns (+9.2%) | 2400 ns | +0 ns (+0.0%) | 2460 ns | +60 ns (+2.5%) | 2400 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.17 % | 0.17 % | +0.00 pp | 0.17 % | +0.00 pp | 0.17 % | +0.00 pp | 0.17 % | +0.00 pp |
| Memory Throughput (%) | 0.50 % | 0.46 % | -0.04 pp | 0.42 % | -0.08 pp | 0.51 % | +0.01 pp | 0.44 % | -0.06 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.18 % | -0.01 pp | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp | 6.20 % | +0.01 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 85.70 % | 85.71 % | +0.01 pp | 85.41 % | -0.29 pp | 88.46 % | +2.76 pp | 86.82 % | +1.12 pp |
| IPC Active | 0.48 inst/cycle | 0.54 inst/cycle | +0.06 | 0.51 inst/cycle | +0.03 | 0.51 inst/cycle | +0.03 | 0.40 inst/cycle | -0.08 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_kernel_with_flat_2d_grid_and_correct_indexing | — | 0.988× | ✓ |  |
| D | fix_grid_launch_3d_matching_kernel_pid_scheme | — | 1.000× | ✓ |  |
| E | add_triton_autotune | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rewrite_kernel_wi... | Δ | fix_grid_launch_3... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2430 ns | +30 ns (+1.2%) | 2400 ns | +0 ns (+0.0%) | 2400 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.17 % | 0.15 % | -0.02 pp | 0.16 % | -0.01 pp | 0.17 % | +0.00 pp |
| Memory Throughput (%) | 0.50 % | 0.41 % | -0.09 pp | 0.42 % | -0.08 pp | 0.44 % | -0.06 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.18 % | -0.01 pp | 6.18 % | -0.01 pp | 6.20 % | +0.01 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 85.70 % | 87.25 % | +1.55 pp | 85.45 % | -0.25 pp | 85.58 % | -0.12 pp |
| IPC Active | 0.48 inst/cycle | 0.47 inst/cycle | -0.01 | 0.54 inst/cycle | +0.06 | 0.43 inst/cycle | -0.05 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | kernel_fusion_correct_3d_grid_parallelism | — | 0.902× | ✓ |  |
| B | fix_kernel_function_grid_to_match_existing_3d_pid_scheme_with_correct_yblock_tiling | — | 1.026× | ✓ |  |
| C | fix_grid_launch_2d_with_correct_xblock_yblock_matching_kernel_pid_scheme | — | 0.949× | ✓ |  |


### Metric Comparison

| Metric | Baseline | kernel_fusion_cor... | Δ | fix_kernel_functi... | Δ | fix_grid_launch_2... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2660 ns | +260 ns (+10.8%) | 2340 ns | -60 ns (-2.5%) | 2530 ns | +130 ns (+5.4%) |
| SM Throughput (%) | 0.17 % | 0.17 % | +0.00 pp | 0.15 % | -0.02 pp | 0.15 % | -0.02 pp |
| Memory Throughput (%) | 0.50 % | 0.40 % | -0.10 pp | 0.42 % | -0.08 pp | 0.43 % | -0.07 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.18 % | -0.01 pp | 6.18 % | -0.01 pp | 6.20 % | +0.01 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 85.70 % | 85.66 % | -0.04 pp | 86.39 % | +0.69 pp | 86.27 % | +0.57 pp |
| IPC Active | 0.48 inst/cycle | 0.53 inst/cycle | +0.05 | 0.50 inst/cycle | +0.02 | 0.34 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  2400 ns  (1.000×)

Final best: 2620 ns  (0.9160×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_clone_permute_slice_unsqueeze_view_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -23,8 +24,6 @@

     y5 = yindex
     tmp0 = tl.load(in_ptr0 + (y0 + 16*x4 + 1024*y2 + 3072*y1 + 24576*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (x4 + 64*y5), tmp0, xmask & ymask)
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
