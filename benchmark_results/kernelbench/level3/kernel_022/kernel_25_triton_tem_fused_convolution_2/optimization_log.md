# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 32540 ns — **Best latency**: 32540 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_25`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_25_triton_tem_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.31
- **Registers/thread**: 128
- **Duration**: 32540 ns
- **SM Throughput**: 18.62 % | **Memory**: 72.55 % | **DRAM**: 5.09 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.993× | 0.991× | ✓ |  |
| B | tma_loads_with_shared_memory_staging_and_autotuning | 0.950× | 0.958× | ✓ |  |
| C | fix_b_matrix_load_coalescing_and_add_autotune | 0.997× | 1.001× | ✓ |  |
| D | add_autotune_reduce_register_pressure_fix_bank_conflicts | 0.992× | 0.704× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_sh... | Δ | fix_b_matrix_load... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32540 ns | 32830 ns | +290 ns (+0.9%) | 33980 ns | +1440 ns (+4.4%) | 32510 ns | -30 ns (-0.1%) | 46240 ns | +13700 ns (+42.1%) |
| SM Throughput (%) | 18.62 % | 18.59 % | -0.03 pp | 17.90 % | -0.72 pp | 18.42 % | -0.20 pp | 12.69 % | -5.93 pp |
| Memory Throughput (%) | 72.55 % | 72.44 % | -0.11 pp | 69.70 % | -2.85 pp | 71.73 % | -0.82 pp | 28.76 % | -43.79 pp |
| DRAM Throughput (%) | 5.09 % | 5.05 % | -0.04 pp | 4.88 % | -0.21 pp | 5.10 % | +0.01 pp | 3.60 % | -1.49 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 96.00 register/thread | -32 | 128.00 register/thread | +0 | 255.00 register/thread | +127 |
| Waves/SM | 0.31 | 0.31 | +0.00 | 0.31 | +0.00 | 0.31 | +0.00 | 0.23 | -0.08 |
| No Eligible (%) | 78.31 % | 78.54 % | +0.23 pp | 79.26 % | +0.95 pp | 78.04 % | -0.27 pp | 69.65 % | -8.66 pp |
| IPC Active | 0.86 inst/cycle | 0.86 inst/cycle | +0.00 | 0.83 inst/cycle | -0.03 | 0.87 inst/cycle | +0.01 | 0.61 inst/cycle | -0.25 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  32540 ns  (1.000×)

Final best: 32540 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
