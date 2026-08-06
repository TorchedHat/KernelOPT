# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_permute_unsqueeze_view_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2370 ns — **Best latency**: 2340 ns (**1.0128×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`rewrite_kernel_2d_grid_correct_yoffset_no_pid2`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_permute_unsqueeze_view_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_9_triton_poi_fused_clone_permute/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 28
- **Duration**: 2370 ns
- **SM Throughput**: 0.11 % | **Memory**: 0.44 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.18 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 88.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_launch_to_expose_2d_parallelism | 1.009× | 0.975× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2370 ns | 2430 ns | +60 ns (+2.5%) |
| SM Throughput (%) | 0.11 % | 0.17 % | +0.06 pp |
| Memory Throughput (%) | 0.44 % | 0.48 % | +0.04 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.05 % | 85.53 % | -2.52 pp |
| IPC Active | 0.56 inst/cycle | 0.54 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_3d_to_match_kernel_pid_scheme | 1.003× | 1.000× | ✓ |  |
| B | fix_3d_grid_launch_with_correct_yblock_tiling | 1.007× | 0.915× | ✓ |  |
| E | add_triton_autotune | 0.982× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_3... | Δ | fix_3d_grid_launc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2370 ns | +0 ns (+0.0%) | 2590 ns | +220 ns (+9.3%) | 2370 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.11 % | 0.16 % | +0.05 pp | 0.17 % | +0.06 pp | 0.15 % | +0.04 pp |
| Memory Throughput (%) | 0.44 % | 0.46 % | +0.02 pp | 0.57 % | +0.13 pp | 0.44 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.05 % | 88.88 % | +0.83 pp | 90.37 % | +2.32 pp | 89.56 % | +1.51 pp |
| IPC Active | 0.56 inst/cycle | 0.57 inst/cycle | +0.01 | 0.53 inst/cycle | -0.03 | 0.56 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_triton_autotune | 1.042× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2370 ns | 2370 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.11 % | 0.12 % | +0.01 pp |
| Memory Throughput (%) | 0.44 % | 0.54 % | +0.10 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.05 % | 85.51 % | -2.54 pp |
| IPC Active | 0.56 inst/cycle | 0.57 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_to_use_autotuned_xblock_yblock | 1.041× | 1.000× | ✓ |  |
| C | fix_grid_launch_correct_3d_with_y_flat_in_dim1 | 0.978× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_launch_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2370 ns | +0 ns (+0.0%) | 2430 ns | +60 ns (+2.5%) |
| SM Throughput (%) | 0.11 % | 0.16 % | +0.05 pp | 0.16 % | +0.05 pp |
| Memory Throughput (%) | 0.44 % | 0.54 % | +0.10 pp | 0.44 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.05 % | 85.65 % | -2.40 pp | 85.45 % | -2.60 pp |
| IPC Active | 0.56 inst/cycle | 0.54 inst/cycle | -0.02 | 0.53 inst/cycle | -0.03 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_kernel_function_3d_grid_correct_y_tile_split | 1.026× | 1.000× | ✓ |  |
| B | fix_grid_launch_2d_simple_pid1_only_yoffset | 1.007× | 0.915× | ✓ |  |
| C | rewrite_kernel_2d_grid_correct_yoffset_no_pid2 | 1.044× | 1.013× | ✓ | **✓ best** |
| D | fix_grid_launch_2d_correct_pid1_only_y_tiling | 1.000× | 0.881× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_kernel_functi... | Δ | fix_grid_launch_2... | Δ | rewrite_kernel_2d... | Δ | fix_grid_launch_2... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2370 ns | 2370 ns | +0 ns (+0.0%) | 2590 ns | +220 ns (+9.3%) | 2340 ns | -30 ns (-1.3%) | 2690 ns | +320 ns (+13.5%) |
| SM Throughput (%) | 0.11 % | 0.15 % | +0.04 pp | 0.16 % | +0.05 pp | 0.16 % | +0.05 pp | 0.17 % | +0.06 pp |
| Memory Throughput (%) | 0.44 % | 0.44 % | +0.00 pp | 0.41 % | -0.03 pp | 0.45 % | +0.01 pp | 0.40 % | -0.04 pp |
| DRAM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp | 0.06 % | -0.01 pp | 0.06 % | -0.01 pp | 0.06 % | -0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 29.00 register/thread | +1 | 29.00 register/thread | +1 | 28.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.05 % | 84.72 % | -3.33 pp | 84.85 % | -3.20 pp | 88.08 % | +0.03 pp | 84.73 % | -3.32 pp |
| IPC Active | 0.56 inst/cycle | 0.57 inst/cycle | +0.01 | 0.54 inst/cycle | -0.02 | 0.53 inst/cycle | -0.03 | 0.55 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  2370 ns  (1.000×)
Iter 5     2340 ns  (1.013×)  — rewrite_kernel_2d_grid_correct_yoffset_no_pid2

Final best: 2340 ns  (1.0128×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,12 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_clone_permute_unsqueeze_view_9(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 2097152
     xnumel = 64
-    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
+    yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
@@ -24,8 +25,6 @@

     tmp0 = tl.load(in_ptr0 + (y4 + 512*x3 + 32768*y2), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 64*x3 + 4096*y5), tmp0, xmask & ymask)
     tl.store(out_ptr1 + (x3 + 64*y6), tmp0, xmask & ymask)
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
