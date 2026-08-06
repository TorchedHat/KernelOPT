# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5730 ns — **Best latency**: 5730 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_11_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 121
- **Duration**: 5730 ns
- **SM Throughput**: 0.21 % | **Memory**: 0.50 % | **DRAM**: 0.34 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 1.50 inst/cycle | **No Eligible**: 61.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_and_full_grid_launch | 0.683× | 0.933× | ✓ |  |
| B | fix_grid_to_cover_full_batch_dimension | 0.432× | 0.933× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | fix_grid_to_cover... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5730 ns | 6140 ns | +410 ns (+7.2%) | 6140 ns | +410 ns (+7.2%) |
| SM Throughput (%) | 0.21 % | 0.21 % | +0.00 pp | 0.21 % | +0.00 pp |
| Memory Throughput (%) | 0.50 % | 0.50 % | +0.00 pp | 2.26 % | +1.76 pp |
| DRAM Throughput (%) | 0.34 % | 0.28 % | -0.06 pp | 0.32 % | -0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 121.00 register/thread | 121.00 register/thread | +0 | 121.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 61.48 % | 59.36 % | -2.12 pp | 59.38 % | -2.10 pp |
| IPC Active | 1.50 inst/cycle | 1.57 inst/cycle | +0.07 | 1.60 inst/cycle | +0.10 |

---

## Cumulative Progress

```
Baseline:  5730 ns  (1.000×)

Final best: 5730 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
