# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6980 ns — **Best latency**: 6980 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_26_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7350,1,1), Waves/SM: 3.48
- **Registers/thread**: 18
- **Duration**: 6980 ns
- **SM Throughput**: 28.49 % | **Memory**: 17.08 % | **DRAM**: 11.05 %
- **Achieved Occupancy**: 49.28 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 59.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.961× | 0.936× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_memory_latency | 0.960× | 0.936× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6980 ns | 7460 ns | +480 ns (+6.9%) | 7460 ns | +480 ns (+6.9%) |
| SM Throughput (%) | 28.49 % | 36.86 % | +8.37 pp | 36.93 % | +8.44 pp |
| Memory Throughput (%) | 17.08 % | 23.72 % | +6.64 pp | 23.63 % | +6.55 pp |
| DRAM Throughput (%) | 11.05 % | 10.34 % | -0.71 pp | 10.35 % | -0.70 pp |
| Achieved Occupancy (%) | 49.28 % | 45.59 % | -3.69 pp | 47.11 % | -2.17 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.48 | 3.48 | +0.00 | 3.48 | +0.00 |
| No Eligible (%) | 59.07 % | 40.44 % | -18.63 pp | 40.13 % | -18.94 pp |
| IPC Active | 1.67 inst/cycle | 2.36 inst/cycle | +0.69 | 2.40 inst/cycle | +0.73 |

---

## Cumulative Progress

```
Baseline:  6980 ns  (1.000×)

Final best: 6980 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
