# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4100 ns — **Best latency**: 4060 ns (**1.0099×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_grid_to_parallelize_all_batch_tiles`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_10_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 56
- **Duration**: 4100 ns
- **SM Throughput**: 0.12 % | **Memory**: 0.42 % | **DRAM**: 0.16 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 1.03 inst/cycle | **No Eligible**: 75.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_and_fix_grid_launch | 1.006× | 0.972× | ✓ |  |
| B | fix_grid_to_parallelize_all_batch_tiles | 1.028× | 1.010× | ✓ |  |
| D | fix_grid_launch_to_cover_full_batch_dimension | 1.035× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | fix_grid_to_paral... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4100 ns | 4220 ns | +120 ns (+2.9%) | 4060 ns | -40 ns (-1.0%) | 4060 ns | -40 ns (-1.0%) |
| SM Throughput (%) | 0.12 % | 0.11 % | -0.01 pp | 0.11 % | -0.01 pp | 0.12 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.40 % | -0.02 pp | 0.41 % | -0.01 pp | 0.46 % | +0.04 pp |
| DRAM Throughput (%) | 0.16 % | 0.16 % | +0.00 pp | 0.16 % | +0.00 pp | 0.16 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 56.00 register/thread | +0 | 56.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.05 % | 74.72 % | -0.33 pp | 76.85 % | +1.80 pp | 76.06 % | +1.01 pp |
| IPC Active | 1.03 inst/cycle | 0.91 inst/cycle | -0.12 | 0.91 inst/cycle | -0.12 | 0.99 inst/cycle | -0.04 |

---

## Iteration 2

**Best so far**: 1.0099× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_batch_grid_dimension_only | 1.001× | 1.000× | ✓ |  |
| B | expand_batch_grid_dimension_without_split_k | 0.998× | 0.956× | ✓ |  |
| C | expand_batch_grid_and_enable_tf32 | 1.031× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_batch_grid... | Δ | expand_batch_grid... | Δ | expand_batch_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4100 ns | 4100 ns | +0 ns (+0.0%) | 4290 ns | +190 ns (+4.6%) | 4130 ns | +30 ns (+0.7%) |
| SM Throughput (%) | 0.12 % | 0.11 % | -0.01 pp | 0.12 % | +0.00 pp | 0.11 % | -0.01 pp |
| Memory Throughput (%) | 0.42 % | 0.40 % | -0.02 pp | 0.46 % | +0.04 pp | 0.45 % | +0.03 pp |
| DRAM Throughput (%) | 0.16 % | 0.16 % | +0.00 pp | 0.15 % | -0.01 pp | 0.16 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 56.00 register/thread | +0 | 56.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.05 % | 76.04 % | +0.99 pp | 76.55 % | +1.50 pp | 74.81 % | -0.24 pp |
| IPC Active | 1.03 inst/cycle | 1.02 inst/cycle | -0.01 | 0.92 inst/cycle | -0.11 | 0.99 inst/cycle | -0.04 |

---

## Iteration 3

**Best so far**: 1.0099× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.980× | 1.017× | ✓ |  |
| B | fix_batch_grid_and_widen_block_n_for_bank_conflict_elimination | 0.958× | 0.878× | ✓ |  |
| C | fix_grid_batch_dimension_with_tf32 | 1.030× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_batch_grid_an... | Δ | fix_grid_batch_di... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4100 ns | 4030 ns | -70 ns (-1.7%) | 4670 ns | +570 ns (+13.9%) | 4100 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.12 % | 0.11 % | -0.01 pp | 0.16 % | +0.04 pp | 0.12 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.42 % | +0.00 pp | 0.42 % | +0.00 pp | 0.40 % | -0.02 pp |
| DRAM Throughput (%) | 0.16 % | 0.16 % | +0.00 pp | 0.16 % | +0.00 pp | 0.16 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.23 % | -0.01 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 72.00 register/thread | +16 | 56.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 75.05 % | 75.10 % | +0.05 pp | 69.15 % | -5.90 pp | 76.78 % | +1.73 pp |
| IPC Active | 1.03 inst/cycle | 1.00 inst/cycle | -0.03 | 1.23 inst/cycle | +0.20 | 1.00 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  4100 ns  (1.000×)
Iter 1     4060 ns  (1.010×)  — fix_grid_launch_to_cover_full_batch_dimension

Final best: 4060 ns  (1.0099×)
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
 def triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10(arg_A, arg_B, out_ptr0):
@@ -86,8 +87,6 @@

     xindex = idx_n + 16*idx_m + 1024*idx_q
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
