# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285700 ns — **Best latency**: 285700 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_069/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285700 ns
- **SM Throughput**: 19.59 % | **Memory**: 78.88 % | **DRAM**: 72.23 %
- **Achieved Occupancy**: 94.01 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_for_coalesced_io | 0.984× | 0.987× | ✓ |  |
| D | increase_tile_sizes_for_cache_reuse_and_parallelism | 0.998× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285700 ns | 289600 ns | +3900 ns (+1.4%) | 286340 ns | +640 ns (+0.2%) |
| SM Throughput (%) | 19.59 % | 18.75 % | -0.84 pp | 19.94 % | +0.35 pp |
| Memory Throughput (%) | 78.88 % | 77.94 % | -0.94 pp | 78.67 % | -0.21 pp |
| DRAM Throughput (%) | 72.23 % | 71.26 % | -0.97 pp | 72.05 % | -0.18 pp |
| Achieved Occupancy (%) | 94.01 % | 93.90 % | -0.11 pp | 94.06 % | +0.05 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.80 % | 81.27 % | +1.47 pp | 79.73 % | -0.07 pp |
| IPC Active | 0.81 inst/cycle | 0.75 inst/cycle | -0.06 | 0.79 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  285700 ns  (1.000×)

Final best: 285700 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
