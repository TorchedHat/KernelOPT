# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7390 ns — **Best latency**: 7390 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_228_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7296,1,1), Waves/SM: 3.45
- **Registers/thread**: 28
- **Duration**: 7390 ns
- **SM Throughput**: 54.37 % | **Memory**: 51.37 % | **DRAM**: 5.83 %
- **Achieved Occupancy**: 70.75 %
- **IPC Active**: 3.11 inst/cycle | **No Eligible**: 21.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_xblock_oversize_and_eliminate_partial_wave | 0.982× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_xblock_oversi... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7390 ns | 7460 ns | +70 ns (+0.9%) |
| SM Throughput (%) | 54.37 % | 46.85 % | -7.52 pp |
| Memory Throughput (%) | 51.37 % | 52.92 % | +1.55 pp |
| DRAM Throughput (%) | 5.83 % | 5.77 % | -0.06 pp |
| Achieved Occupancy (%) | 70.75 % | 71.27 % | +0.52 pp |
| Registers/Thread | 28.00 register/thread | 29.00 register/thread | +1 |
| Waves/SM | 3.45 | 1.73 | -1.72 |
| No Eligible (%) | 21.55 % | 27.12 % | +5.57 pp |
| IPC Active | 3.11 inst/cycle | 2.88 inst/cycle | -0.23 |

---

## Cumulative Progress

```
Baseline:  7390 ns  (1.000×)

Final best: 7390 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
