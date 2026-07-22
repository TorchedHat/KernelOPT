# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45440 ns — **Best latency**: 45440 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_5_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 4.95
- **Registers/thread**: 168
- **Duration**: 45440 ns
- **SM Throughput**: 30.56 % | **Memory**: 77.14 % | **DRAM**: 13.14 %
- **Achieved Occupancy**: 17.92 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 66.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.001× | ✓ |  |
| B | reduce_register_pressure_and_shared_memory_conflicts_via_tile_reshape | 0.977× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45440 ns | 45380 ns | -60 ns (-0.1%) | 46050 ns | +610 ns (+1.3%) |
| SM Throughput (%) | 30.56 % | 30.58 % | +0.02 pp | 29.84 % | -0.72 pp |
| Memory Throughput (%) | 77.14 % | 77.19 % | +0.05 pp | 76.18 % | -0.96 pp |
| DRAM Throughput (%) | 13.14 % | 13.25 % | +0.11 pp | 12.94 % | -0.20 pp |
| Achieved Occupancy (%) | 17.92 % | 17.91 % | -0.01 pp | 17.95 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.63 % | 66.68 % | +0.05 pp | 67.40 % | +0.77 pp |
| IPC Active | 1.32 inst/cycle | 1.33 inst/cycle | +0.01 | 1.30 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  45440 ns  (1.000×)

Final best: 45440 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
