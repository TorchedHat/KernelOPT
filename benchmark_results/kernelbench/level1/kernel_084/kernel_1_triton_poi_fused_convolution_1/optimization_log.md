# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286270 ns — **Best latency**: 286270 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_084/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286270 ns
- **SM Throughput**: 19.70 % | **Memory**: 78.78 % | **DRAM**: 72.08 %
- **Achieved Occupancy**: 94.01 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | increase_yblock_to_match_channel_dimension | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_yblock_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286270 ns | 286500 ns | +230 ns (+0.1%) |
| SM Throughput (%) | 19.70 % | 19.72 % | +0.02 pp |
| Memory Throughput (%) | 78.78 % | 78.68 % | -0.10 pp |
| DRAM Throughput (%) | 72.08 % | 72.04 % | -0.04 pp |
| Achieved Occupancy (%) | 94.01 % | 94.02 % | +0.01 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.97 % | 80.34 % | +0.37 pp |
| IPC Active | 0.81 inst/cycle | 0.79 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  286270 ns  (1.000×)

Final best: 286270 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
