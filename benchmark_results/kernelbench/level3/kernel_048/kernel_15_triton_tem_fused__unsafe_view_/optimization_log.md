# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3840 ns — **Best latency**: 3740 ns (**1.0267×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_grid_launch_with_correct_multidim_batch_offset`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_15_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 80
- **Duration**: 3840 ns
- **SM Throughput**: 0.10 % | **Memory**: 0.48 % | **DRAM**: 0.12 %
- **Achieved Occupancy**: 6.20 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 79.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_and_fix_grid_launch | — | 1.027× | ✓ | **✓ best** |
| C | fix_grid_to_cover_full_batch_dimension | — | 1.027× | ✓ |  |
| D | fix_grid_launch_and_autotune_tile_sizes | — | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | fix_grid_to_cover... | Δ | fix_grid_launch_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 3740 ns | -100 ns (-2.6%) | 3740 ns | -100 ns (-2.6%) | 3780 ns | -60 ns (-1.6%) |
| SM Throughput (%) | 0.10 % | 0.10 % | +0.00 pp | 0.10 % | +0.00 pp | 0.10 % | +0.00 pp |
| Memory Throughput (%) | 0.48 % | 0.56 % | +0.08 pp | 0.49 % | +0.01 pp | 0.49 % | +0.01 pp |
| DRAM Throughput (%) | 0.12 % | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 79.86 % | 76.81 % | -3.05 pp | 77.74 % | -2.12 pp | 76.72 % | -3.14 pp |
| IPC Active | 0.86 inst/cycle | 0.90 inst/cycle | +0.04 | 0.80 inst/cycle | -0.06 | 0.83 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0267× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tf32_tensor_cores | — | 0.953× | ✓ |  |
| C | fix_grid_launch_with_tf32_tensor_cores | — | 0.992× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 4030 ns | +190 ns (+4.9%) | 3870 ns | +30 ns (+0.8%) |
| SM Throughput (%) | 0.10 % | 0.10 % | +0.00 pp | 0.10 % | +0.00 pp |
| Memory Throughput (%) | 0.48 % | 0.49 % | +0.01 pp | 0.58 % | +0.10 pp |
| DRAM Throughput (%) | 0.12 % | 0.11 % | -0.01 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 79.86 % | 77.53 % | -2.33 pp | 76.51 % | -3.35 pp |
| IPC Active | 0.86 inst/cycle | 0.88 inst/cycle | +0.02 | 0.89 inst/cycle | +0.03 |

---

## Iteration 3

**Best so far**: 1.0267× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_only_preserve_fp32 | — | 0.967× | ✓ |  |
| C | fix_grid_launch_with_correct_multidim_batch_offset | — | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_o... | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 3970 ns | +130 ns (+3.4%) | 3740 ns | -100 ns (-2.6%) |
| SM Throughput (%) | 0.10 % | 0.09 % | -0.01 pp | 0.10 % | +0.00 pp |
| Memory Throughput (%) | 0.48 % | 0.54 % | +0.06 pp | 0.50 % | +0.02 pp |
| DRAM Throughput (%) | 0.12 % | 0.12 % | +0.00 pp | 0.12 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.20 % | +0.00 pp | 6.20 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 80.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 79.86 % | 77.29 % | -2.57 pp | 77.50 % | -2.36 pp |
| IPC Active | 0.86 inst/cycle | 0.81 inst/cycle | -0.05 | 0.88 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  3840 ns  (1.000×)
Iter 1     3740 ns  (1.027×)  — tf32_tensor_cores_and_fix_grid_launch

Final best: 3740 ns  (1.0267×)
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
 def triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15(arg_A, arg_B, out_ptr0):
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

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
