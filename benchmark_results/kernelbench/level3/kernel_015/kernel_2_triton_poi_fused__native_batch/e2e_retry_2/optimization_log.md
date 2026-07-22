# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21860 ns — **Best latency**: 21860 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 16
- **Duration**: 21860 ns
- **SM Throughput**: 35.48 % | **Memory**: 23.46 % | **DRAM**: 14.97 %
- **Achieved Occupancy**: 40.47 %
- **IPC Active**: 1.52 inst/cycle | **No Eligible**: 61.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_hide_l1tex_latency | 0.997× | 1.009× | ✓ |  |
| B | autotune_xblock_to_hide_l1tex_latency | 0.910× | 0.945× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 21660 ns | -200 ns (-0.9%) | 23140 ns | +1280 ns (+5.9%) |
| SM Throughput (%) | 35.48 % | 35.42 % | -0.06 pp | 58.28 % | +22.80 pp |
| Memory Throughput (%) | 23.46 % | 23.40 % | -0.06 pp | 34.65 % | +11.19 pp |
| DRAM Throughput (%) | 14.97 % | 15.10 % | +0.13 pp | 14.15 % | -0.82 pp |
| Achieved Occupancy (%) | 40.47 % | 40.87 % | +0.40 pp | 41.93 % | +1.46 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 32.00 register/thread | +16 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 61.72 % | 61.63 % | -0.09 pp | 32.95 % | -28.77 pp |
| IPC Active | 1.52 inst/cycle | 1.53 inst/cycle | +0.01 | 2.67 inst/cycle | +1.15 |

---

## Cumulative Progress

```
Baseline:  21860 ns  (1.000×)

Final best: 21860 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
