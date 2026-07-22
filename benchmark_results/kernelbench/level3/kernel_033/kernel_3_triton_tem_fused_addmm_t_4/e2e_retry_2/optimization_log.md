# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4010000 ns — **Best latency**: 4010000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_3_triton_tem_fused_addmm_t_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 4010000 ns
- **SM Throughput**: 37.68 % | **Memory**: 85.99 % | **DRAM**: 3.14 %
- **Achieved Occupancy**: 12.25 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotune_to_fix_register_pressure_and_bank_conflicts | 0.987× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4010000 ns | 4060000 ns | +50000 ns (+1.2%) |
| SM Throughput (%) | 37.68 % | 37.17 % | -0.51 pp |
| Memory Throughput (%) | 85.99 % | 84.34 % | -1.65 pp |
| DRAM Throughput (%) | 3.14 % | 3.08 % | -0.06 pp |
| Achieved Occupancy (%) | 12.25 % | 12.24 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 61.24 % | 61.81 % | +0.57 pp |
| IPC Active | 1.55 inst/cycle | 1.53 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  4010000 ns  (1.000×)

Final best: 4010000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
