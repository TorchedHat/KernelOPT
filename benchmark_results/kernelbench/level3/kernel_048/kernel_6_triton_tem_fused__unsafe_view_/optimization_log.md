# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3780 ns — **Best latency**: 3840 ns (**0.9844×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_6_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 80
- **Duration**: 3780 ns
- **SM Throughput**: 0.10 % | **Memory**: 0.53 % | **DRAM**: 0.12 %
- **Achieved Occupancy**: 6.20 %
- **IPC Active**: 0.84 inst/cycle | **No Eligible**: 78.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 0.984× | ✓ | **✓ best** |
| B | fix_grid_launch_and_autotune_bmm | 0.969× | 0.969× | ✓ |  |
| C | fix_grid_to_cover_all_batch_tiles | 1.000× | 1.000× | ✓ |  |
| D | fix_grid_to_cover_full_batch_dimension | 0.960× | 1.019× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_grid_launch_a... | Δ | fix_grid_to_cover... | Δ | fix_grid_to_cover... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3780 ns | 3840 ns | +60 ns (+1.6%) | 3900 ns | +120 ns (+3.2%) | 3780 ns | +0 ns (+0.0%) | 3710 ns | -70 ns (-1.9%) |
| SM Throughput (%) | 0.10 % | 0.10 % | +0.00 pp | 0.10 % | +0.00 pp | 0.09 % | -0.01 pp | 0.10 % | +0.00 pp |
| Memory Throughput (%) | 0.53 % | 0.47 % | -0.06 pp | 0.51 % | -0.02 pp | 0.54 % | +0.01 pp | 0.49 % | -0.04 pp |
| DRAM Throughput (%) | 0.12 % | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 78.69 % | 77.45 % | -1.24 pp | 77.32 % | -1.37 pp | 80.52 % | +1.83 pp | 78.18 % | -0.51 pp |
| IPC Active | 0.84 inst/cycle | 0.88 inst/cycle | +0.04 | 0.89 inst/cycle | +0.05 | 0.88 inst/cycle | +0.04 | 0.88 inst/cycle | +0.04 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_cover_full_batch_with_autotune | 0.957× | 0.977× | ✓ |  |
| B | autotune_with_correct_grid_and_reduced_register_pressure | 0.996× | 1.011× | ✓ |  |
| C | fix_grid_to_cover_all_batch_elements_with_autotune | 0.995× | 1.019× | ✓ |  |
| D | fix_grid_to_cover_all_batch_slices | 0.996× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_cover... | Δ | autotune_with_cor... | Δ | fix_grid_to_cover... | Δ | fix_grid_to_cover... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3780 ns | 3870 ns | +90 ns (+2.4%) | 3740 ns | -40 ns (-1.1%) | 3710 ns | -70 ns (-1.9%) | 3740 ns | -40 ns (-1.1%) |
| SM Throughput (%) | 0.10 % | 0.10 % | +0.00 pp | 0.09 % | -0.01 pp | 0.10 % | +0.00 pp | 0.09 % | -0.01 pp |
| Memory Throughput (%) | 0.53 % | 0.62 % | +0.09 pp | 0.57 % | +0.04 pp | 0.49 % | -0.04 pp | 0.48 % | -0.05 pp |
| DRAM Throughput (%) | 0.12 % | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 78.69 % | 78.24 % | -0.45 pp | 79.86 % | +1.17 pp | 77.34 % | -1.35 pp | 77.62 % | -1.07 pp |
| IPC Active | 0.84 inst/cycle | 0.88 inst/cycle | +0.04 | 0.89 inst/cycle | +0.05 | 0.81 inst/cycle | -0.03 | 0.89 inst/cycle | +0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_and_reduce_register_pressure_with_autotune | 0.993× | 1.011× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tiles_and_autotune | 0.996× | 0.922× | ✓ |  |
| D | reduce_register_pressure_with_maxnreg | 0.964× | 0.922× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_and_redu... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3780 ns | 3740 ns | -40 ns (-1.1%) | 4100 ns | +320 ns (+8.5%) | 4100 ns | +320 ns (+8.5%) |
| SM Throughput (%) | 0.10 % | 0.10 % | +0.00 pp | 0.09 % | -0.01 pp | 0.10 % | +0.00 pp |
| Memory Throughput (%) | 0.53 % | 0.53 % | +0.00 pp | 0.45 % | -0.08 pp | 0.48 % | -0.05 pp |
| DRAM Throughput (%) | 0.12 % | 0.12 % | +0.00 pp | 0.11 % | -0.01 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.21 % | +0.01 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 64.00 register/thread | -16 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 78.69 % | 77.52 % | -1.17 pp | 77.61 % | -1.08 pp | 77.83 % | -0.86 pp |
| IPC Active | 0.84 inst/cycle | 0.81 inst/cycle | -0.03 | 0.88 inst/cycle | +0.04 | 0.82 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  3780 ns  (1.000×)

Final best: 3840 ns  (0.9844×)
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
 def triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6(arg_A, arg_B, out_ptr0):
@@ -86,8 +87,6 @@

     xindex = idx_n + 64*idx_m + 4096*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
