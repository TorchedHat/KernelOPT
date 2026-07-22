# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4380 ns — **Best latency**: 4380 ns (**1.0000×** vs baseline)
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
- **Duration**: 4380 ns
- **SM Throughput**: 25.98 % | **Memory**: 9.40 % | **DRAM**: 6.22 %
- **Achieved Occupancy**: 59.36 %
- **IPC Active**: 2.05 inst/cycle | **No Eligible**: 50.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave | 0.954× | 0.950× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.985× | 0.950× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4380 ns | 4610 ns | +230 ns (+5.3%) | 4610 ns | +230 ns (+5.3%) |
| SM Throughput (%) | 25.98 % | 43.94 % | +17.96 pp | 43.03 % | +17.05 pp |
| Memory Throughput (%) | 9.40 % | 12.06 % | +2.66 pp | 11.79 % | +2.39 pp |
| DRAM Throughput (%) | 6.22 % | 5.95 % | -0.27 pp | 5.91 % | -0.31 pp |
| Achieved Occupancy (%) | 59.36 % | 70.44 % | +11.08 pp | 70.21 % | +10.85 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.22 | 1.22 | +0.00 | 1.22 | +0.00 |
| No Eligible (%) | 50.80 % | 23.34 % | -27.46 pp | 25.82 % | -24.98 pp |
| IPC Active | 2.05 inst/cycle | 3.00 inst/cycle | +0.95 | 3.01 inst/cycle | +0.96 |

---

## Cumulative Progress

```
Baseline:  4380 ns  (1.000×)

Final best: 4380 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
