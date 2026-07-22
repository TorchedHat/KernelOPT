# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4320 ns — **Best latency**: 4320 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_35_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2573,1,1), Waves/SM: 1.22
- **Registers/thread**: 18
- **Duration**: 4320 ns
- **SM Throughput**: 25.76 % | **Memory**: 9.42 % | **DRAM**: 6.36 %
- **Achieved Occupancy**: 62.08 %
- **IPC Active**: 2.13 inst/cycle | **No Eligible**: 48.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.955× | 0.937× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4320 ns | 4610 ns | +290 ns (+6.7%) |
| SM Throughput (%) | 25.76 % | 41.78 % | +16.02 pp |
| Memory Throughput (%) | 9.42 % | 11.49 % | +2.07 pp |
| DRAM Throughput (%) | 6.36 % | 5.95 % | -0.41 pp |
| Achieved Occupancy (%) | 62.08 % | 70.27 % | +8.19 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 |
| Waves/SM | 1.22 | 1.22 | +0.00 |
| No Eligible (%) | 48.89 % | 25.77 % | -23.12 pp |
| IPC Active | 2.13 inst/cycle | 3.01 inst/cycle | +0.88 |

---

## Cumulative Progress

```
Baseline:  4320 ns  (1.000×)

Final best: 4320 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
