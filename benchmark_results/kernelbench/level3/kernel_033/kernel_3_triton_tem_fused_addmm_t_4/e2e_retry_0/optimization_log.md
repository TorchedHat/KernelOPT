# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4000000 ns — **Best latency**: 4000000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_3_triton_tem_fused_addmm_t_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 4000000 ns
- **SM Throughput**: 37.74 % | **Memory**: 86.15 % | **DRAM**: 3.15 %
- **Achieved Occupancy**: 12.25 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.998× | ✓ |  |
| B | tma_loads_block_ptr_reduce_register_pressure | 0.701× | 0.699× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_block_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4000000 ns | 4010000 ns | +10000 ns (+0.2%) | 5720000 ns | +1720000 ns (+43.0%) |
| SM Throughput (%) | 37.74 % | 37.72 % | -0.02 pp | 26.37 % | -11.37 pp |
| Memory Throughput (%) | 86.15 % | 86.09 % | -0.06 pp | 59.84 % | -26.31 pp |
| DRAM Throughput (%) | 3.15 % | 3.15 % | +0.00 pp | 2.34 % | -0.81 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 6.25 % | -6.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 1.94 | +0.97 |
| No Eligible (%) | 61.25 % | 61.24 % | -0.01 pp | 69.82 % | +8.57 pp |
| IPC Active | 1.55 inst/cycle | 1.55 inst/cycle | +0.00 | 1.21 inst/cycle | -0.34 |

---

## Cumulative Progress

```
Baseline:  4000000 ns  (1.000×)

Final best: 4000000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
