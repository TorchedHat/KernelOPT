# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285890 ns — **Best latency**: 285890 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_050/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285890 ns
- **SM Throughput**: 20.02 % | **Memory**: 78.86 % | **DRAM**: 72.17 %
- **Achieved Occupancy**: 94.03 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 79.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | increase_yblock_to_cover_full_channel_dimension_for_coalesced_loads | 0.690× | 0.697× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_yblock_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285890 ns | 409980 ns | +124090 ns (+43.4%) |
| SM Throughput (%) | 20.02 % | 13.51 % | -6.51 pp |
| Memory Throughput (%) | 78.86 % | 60.19 % | -18.67 pp |
| DRAM Throughput (%) | 72.17 % | 57.76 % | -14.41 pp |
| Achieved Occupancy (%) | 94.03 % | 94.88 % | +0.85 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.71 % | 86.18 % | +6.47 pp |
| IPC Active | 0.79 inst/cycle | 0.55 inst/cycle | -0.24 |

---

## Cumulative Progress

```
Baseline:  285890 ns  (1.000×)

Final best: 285890 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
