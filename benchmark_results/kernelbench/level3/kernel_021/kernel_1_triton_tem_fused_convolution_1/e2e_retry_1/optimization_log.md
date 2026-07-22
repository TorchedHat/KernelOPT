# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 829760 ns — **Best latency**: 829760 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_1_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 165.49
- **Registers/thread**: 160
- **Duration**: 829760 ns
- **SM Throughput**: 25.63 % | **Memory**: 83.31 % | **DRAM**: 6.84 %
- **Achieved Occupancy**: 18.04 %
- **IPC Active**: 1.03 inst/cycle | **No Eligible**: 74.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 829760 ns | 825820 ns | -3940 ns (-0.5%) |
| SM Throughput (%) | 25.63 % | 25.63 % | +0.00 pp |
| Memory Throughput (%) | 83.31 % | 83.30 % | -0.01 pp |
| DRAM Throughput (%) | 6.84 % | 6.88 % | +0.04 pp |
| Achieved Occupancy (%) | 18.04 % | 18.03 % | -0.01 pp |
| Registers/Thread | 160.00 register/thread | 160.00 register/thread | +0 |
| Waves/SM | 165.49 | 165.49 | +0.00 |
| No Eligible (%) | 74.27 % | 74.27 % | +0.00 pp |
| IPC Active | 1.03 inst/cycle | 1.03 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  829760 ns  (1.000×)

Final best: 829760 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
