# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_bias_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 607140 ns — **Best latency**: 607140 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_bias_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_011/synthesis/kernel_10_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 14.85
- **Registers/thread**: 255
- **Duration**: 607140 ns
- **SM Throughput**: 32.33 % | **Memory**: 22.86 % | **DRAM**: 3.77 %
- **Achieved Occupancy**: 12.47 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 67.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | reduce_tile_sizes_to_cut_register_pressure_and_improve_occupancy | 0.369× | 0.902× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 607140 ns | 673090 ns | +65950 ns (+10.9%) |
| SM Throughput (%) | 32.33 % | 59.65 % | +27.32 pp |
| Memory Throughput (%) | 22.86 % | 62.41 % | +39.55 pp |
| DRAM Throughput (%) | 3.77 % | 3.46 % | -0.31 pp |
| Achieved Occupancy (%) | 12.47 % | 24.55 % | +12.08 pp |
| Registers/Thread | 255.00 register/thread | 108.00 register/thread | -147 |
| Waves/SM | 14.85 | 59.39 | +44.54 |
| No Eligible (%) | 67.02 % | 39.46 % | -27.56 pp |
| IPC Active | 1.32 inst/cycle | 2.40 inst/cycle | +1.08 |

---

## Cumulative Progress

```
Baseline:  607140 ns  (1.000×)

Final best: 607140 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
