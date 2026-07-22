# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174460 ns — **Best latency**: 174460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_057/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174460 ns
- **SM Throughput**: 28.27 % | **Memory**: 81.54 % | **DRAM**: 67.32 %
- **Achieved Occupancy**: 94.18 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_corner_turning_transpose | 0.998× | 0.986× | ✓ |  |
| E | add_triton_autotune | 0.999× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_cor... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174460 ns | 177020 ns | +2560 ns (+1.5%) | 175170 ns | +710 ns (+0.4%) |
| SM Throughput (%) | 28.27 % | 26.20 % | -2.07 pp | 27.99 % | -0.28 pp |
| Memory Throughput (%) | 81.54 % | 80.35 % | -1.19 pp | 81.31 % | -0.23 pp |
| DRAM Throughput (%) | 67.32 % | 66.37 % | -0.95 pp | 67.04 % | -0.28 pp |
| Achieved Occupancy (%) | 94.18 % | 94.13 % | -0.05 pp | 94.20 % | +0.02 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.59 % | 72.86 % | +1.27 pp | 71.06 % | -0.53 pp |
| IPC Active | 1.13 inst/cycle | 1.08 inst/cycle | -0.05 | 1.15 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  174460 ns  (1.000×)

Final best: 174460 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
