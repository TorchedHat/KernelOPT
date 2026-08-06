# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1380000 ns — **Best latency**: 1380000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_0_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 1380000 ns
- **SM Throughput**: 27.52 % | **Memory**: 62.04 % | **DRAM**: 1.04 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.23 inst/cycle | **No Eligible**: 69.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 0.919× | 0.920× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1380000 ns | 1500000 ns | +120000 ns (+8.7%) |
| SM Throughput (%) | 27.52 % | 25.25 % | -2.27 pp |
| Memory Throughput (%) | 62.04 % | 57.62 % | -4.42 pp |
| DRAM Throughput (%) | 1.04 % | 0.98 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 |
| Waves/SM | 0.32 | 0.48 | +0.16 |
| No Eligible (%) | 69.28 % | 70.99 % | +1.71 pp |
| IPC Active | 1.23 inst/cycle | 1.16 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  1380000 ns  (1.000×)

Final best: 1380000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
