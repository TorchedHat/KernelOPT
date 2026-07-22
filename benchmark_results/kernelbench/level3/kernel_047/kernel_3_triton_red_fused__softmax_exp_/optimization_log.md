# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_exp_sub_sum_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10110 ns — **Best latency**: 10110 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_exp_sub_sum_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_047/kernel_3_triton_red_fused__softmax_exp_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 32
- **Duration**: 10110 ns
- **SM Throughput**: 9.89 % | **Memory**: 53.09 % | **DRAM**: 53.09 %
- **Achieved Occupancy**: 23.78 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 87.17 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_tile_rebalancing | — | fail | ✓ |  |
| B | memory_coalescing_via_tile_reorder_and_autotune | 0.988× | 0.808× | ✓ |  |
| C | software_pipelining_with_autotune_tile_reshape | — | fail | ✓ |  |
| D | autotune_xblock_r0block_for_occupancy | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 10110 ns | 12510 ns | +2400 ns (+23.7%) |
| SM Throughput (%) | 9.89 % | 18.73 % | +8.84 pp |
| Memory Throughput (%) | 53.09 % | 42.81 % | -10.28 pp |
| DRAM Throughput (%) | 53.09 % | 42.81 % | -10.28 pp |
| Achieved Occupancy (%) | 23.78 % | 14.83 % | -8.95 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 |
| No Eligible (%) | 87.17 % | 76.26 % | -10.91 pp |
| IPC Active | 0.51 inst/cycle | 0.96 inst/cycle | +0.45 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints_and_fixed_autotune_grid | 0.975× | 0.823× | ✓ |  |
| B | fix_autotune_grid_lambda_and_coalesce_inner_loop_access | — | fail | ✓ |  |
| C | software_pipelining_with_corrected_grid_lambda | — | fail | ✓ |  |
| D | fix_autotune_grid_lambda_and_increase_tile_parallelism | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 10110 ns | 12290 ns | +2180 ns (+21.6%) |
| SM Throughput (%) | 9.89 % | 41.65 % | +31.76 pp |
| Memory Throughput (%) | 53.09 % | 43.79 % | -9.30 pp |
| DRAM Throughput (%) | 53.09 % | 43.79 % | -9.30 pp |
| Achieved Occupancy (%) | 23.78 % | 11.95 % | -11.83 pp |
| Registers/Thread | 32.00 register/thread | 192.00 register/thread | +160 |
| Waves/SM | 0.24 | 1.94 | +1.70 |
| No Eligible (%) | 87.17 % | 48.13 % | -39.04 pp |
| IPC Active | 0.51 inst/cycle | 2.13 inst/cycle | +1.62 |

---

## Cumulative Progress

```
Baseline:  10110 ns  (1.000×)

Final best: 10110 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
