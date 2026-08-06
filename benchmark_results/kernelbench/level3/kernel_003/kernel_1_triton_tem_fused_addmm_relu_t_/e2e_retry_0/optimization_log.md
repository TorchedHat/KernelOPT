# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 176740 ns — **Best latency**: 176740 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_1_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 176740 ns
- **SM Throughput**: 26.95 % | **Memory**: 60.74 % | **DRAM**: 0.97 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 69.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 176740 ns | 176800 ns | +60 ns (+0.0%) |
| SM Throughput (%) | 26.95 % | 27.01 % | +0.06 pp |
| Memory Throughput (%) | 60.74 % | 60.86 % | +0.12 pp |
| DRAM Throughput (%) | 0.97 % | 0.97 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 |
| No Eligible (%) | 69.53 % | 69.55 % | +0.02 pp |
| IPC Active | 1.22 inst/cycle | 1.22 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  176740 ns  (1.000×)

Final best: 176740 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
