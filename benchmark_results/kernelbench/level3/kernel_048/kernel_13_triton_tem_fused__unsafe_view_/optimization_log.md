# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2850 ns — **Best latency**: 2690 ns (**1.0595×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_13_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.02
- **Registers/thread**: 32
- **Duration**: 2850 ns
- **SM Throughput**: 1.22 % | **Memory**: 1.39 % | **DRAM**: 0.13 %
- **Achieved Occupancy**: 6.11 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 85.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | — | 1.060× | ✓ | **✓ best** |
| B | expand_batch_grid_dimension_to_cover_all_bmm_batches | — | 1.060× | ✓ |  |
| C | split_k_and_expose_batch_parallelism | — | 1.060× | ✓ |  |
| D | autotune_tile_sizes_and_fix_grid_utilization | — | 0.956× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | expand_batch_grid... | Δ | split_k_and_expos... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2850 ns | 2690 ns | -160 ns (-5.6%) | 2690 ns | -160 ns (-5.6%) | 2690 ns | -160 ns (-5.6%) | 2980 ns | +130 ns (+4.6%) |
| SM Throughput (%) | 1.22 % | 1.31 % | +0.09 pp | 1.33 % | +0.11 pp | 0.68 % | -0.54 pp | 1.81 % | +0.59 pp |
| Memory Throughput (%) | 1.39 % | 1.48 % | +0.09 pp | 1.51 % | +0.12 pp | 0.85 % | -0.54 pp | 2.06 % | +0.67 pp |
| DRAM Throughput (%) | 0.13 % | 0.14 % | +0.01 pp | 0.14 % | +0.01 pp | 0.14 % | +0.01 pp | 0.14 % | +0.01 pp |
| Achieved Occupancy (%) | 6.11 % | 6.11 % | +0.00 pp | 6.11 % | +0.00 pp | 3.08 % | -3.03 pp | 6.21 % | +0.10 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 40.00 register/thread | +8 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.01 | -0.01 | 0.02 | +0.00 |
| No Eligible (%) | 85.83 % | 84.92 % | -0.91 pp | 84.91 % | -0.92 pp | 83.29 % | -2.54 pp | 80.18 % | -5.65 pp |
| IPC Active | 0.56 inst/cycle | 0.57 inst/cycle | +0.01 | 0.57 inst/cycle | +0.01 | 0.31 inst/cycle | -0.25 | 0.74 inst/cycle | +0.18 |

---

## Iteration 2

**Best so far**: 1.0595× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | flatten_batch_and_tile_into_single_grid_dimension | — | 1.088× | ✓ |  |


### Metric Comparison

| Metric | Baseline | flatten_batch_and... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2850 ns | 2620 ns | -230 ns (-8.1%) |
| SM Throughput (%) | 1.22 % | 1.13 % | -0.09 pp |
| Memory Throughput (%) | 1.39 % | 1.49 % | +0.10 pp |
| DRAM Throughput (%) | 0.13 % | 0.14 % | +0.01 pp |
| Achieved Occupancy (%) | 6.11 % | 6.10 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.02 | +0.00 |
| No Eligible (%) | 85.83 % | 87.41 % | +1.58 pp |
| IPC Active | 0.56 inst/cycle | 0.52 inst/cycle | -0.04 |


### Key Learnings

#### ✅ Plan A: `flatten_batch_and_tile_into_single_grid_dimension` (1.088×)

When a Triton BMM kernel uses tl.program_id(1) as the batch index but grid[1] cannot be expanded (e.g., due to framework constraints), encoding both the tile index and batch index into a single flattened program_id(0) — setting grid[0] = num_tiles * num_batches and decoding idx_q = pid_flat // num_tiles; pid = pid_flat % num_tiles — exposes full batch parallelism through a single grid dimension. In this case, the approach reduced elapsed cycles from 4001 to 3916 and executed instructions from 28224 to 24256 (~14% fewer), yielding an 8.78% speedup. The L2 hit rate improved from 44.5% to 49.2%, suggesting better cache reuse from the reordered access pattern introduced by the pid decode.

> **Applicable when**: Triton BMM kernel uses tl.program_id(1) as batch index but grid[1] is hardcoded to 1 and cannot be directly expanded; profiling shows grid_size << sm_count, waves_per_sm < 0.05, achieved occupancy < 10%, 'No Eligible' scheduler slots > 80%; the batch count is known at launch time and num_tiles is a small constant.

> **Avoid when**: Grid[1] can be directly set to the batch count (prefer the simpler expand_batch_grid_dimension fix); num_tiles is not a compile-time constant (decode arithmetic becomes expensive); the kernel is already grid-saturated (waves_per_sm >= 1.0); the flattened grid exceeds hardware limits (max grid[0] = 2^31-1 on CUDA).

---

## Iteration 3

**Best so far**: 1.0595× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  2850 ns  (1.000×)
Iter 1     2690 ns  (1.060×)  — tf32_tensor_cores

Final best: 2690 ns  (1.0595×)
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
 def triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13(arg_A, arg_B, out_ptr0):
@@ -86,8 +87,6 @@

     xindex = idx_n + 1024*idx_m + 3072*idx_q
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
