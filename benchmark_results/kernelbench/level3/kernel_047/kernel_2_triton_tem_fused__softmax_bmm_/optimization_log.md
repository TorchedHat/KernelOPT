# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__softmax_bmm_exp_sub_transpose_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8480 ns — **Best latency**: 8480 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__softmax_bmm_exp_sub_transpose_view_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_047/kernel_2_triton_tem_fused__softmax_bmm_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.01
- **Registers/thread**: 96
- **Duration**: 8480 ns
- **SM Throughput**: 1.24 % | **Memory**: 1.20 % | **DRAM**: 0.59 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 2.11 inst/cycle | **No Eligible**: 48.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_batch_grid_dimension_to_saturate_sms | 0.997× | 0.953× | ✓ |  |
| C | expand_grid_to_parallelize_batch_dimension | 0.995× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_batch_grid_di... | Δ | expand_grid_to_pa... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8480 ns | 8900 ns | +420 ns (+5.0%) | 8670 ns | +190 ns (+2.2%) |
| SM Throughput (%) | 1.24 % | 1.22 % | -0.02 pp | 1.20 % | -0.04 pp |
| Memory Throughput (%) | 1.20 % | 1.19 % | -0.01 pp | 1.16 % | -0.04 pp |
| DRAM Throughput (%) | 0.59 % | 0.56 % | -0.03 pp | 0.58 % | -0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 48.85 % | 46.70 % | -2.15 pp | 48.17 % | -0.68 pp |
| IPC Active | 2.11 inst/cycle | 1.96 inst/cycle | -0.15 | 1.95 inst/cycle | -0.16 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_batch_dim_and_even_k_no_autotune | 1.003× | 0.993× | ✓ |  |
| C | fix_grid_batch_dimension_minimal_safe | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_batch_di... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8480 ns | 8540 ns | +60 ns (+0.7%) |
| SM Throughput (%) | 1.24 % | 1.19 % | -0.05 pp |
| Memory Throughput (%) | 1.20 % | 1.16 % | -0.04 pp |
| DRAM Throughput (%) | 0.59 % | 0.59 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 48.85 % | 47.22 % | -1.63 pp |
| IPC Active | 2.11 inst/cycle | 2.05 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  8480 ns  (1.000×)

Final best: 8480 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
