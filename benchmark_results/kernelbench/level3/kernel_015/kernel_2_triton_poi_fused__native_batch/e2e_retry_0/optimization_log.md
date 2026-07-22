# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21660 ns — **Best latency**: 21660 ns (**1.0000×** vs baseline)
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
- **Duration**: 21660 ns
- **SM Throughput**: 34.81 % | **Memory**: 23.01 % | **DRAM**: 15.11 %
- **Achieved Occupancy**: 40.69 %
- **IPC Active**: 1.53 inst/cycle | **No Eligible**: 61.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_hide_l1tex_latency | 0.909× | 0.940× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.996× | 0.936× | ✓ |  |
| C | add_triton_autotune | 0.995× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21660 ns | 23040 ns | +1380 ns (+6.4%) | 23140 ns | +1480 ns (+6.8%) | 21700 ns | +40 ns (+0.2%) |
| SM Throughput (%) | 34.81 % | 58.39 % | +23.58 pp | 58.29 % | +23.48 pp | 42.64 % | +7.83 pp |
| Memory Throughput (%) | 23.01 % | 34.62 % | +11.61 pp | 34.42 % | +11.41 pp | 23.67 % | +0.66 pp |
| DRAM Throughput (%) | 15.11 % | 14.21 % | -0.90 pp | 14.16 % | -0.95 pp | 15.09 % | -0.02 pp |
| Achieved Occupancy (%) | 40.69 % | 42.46 % | +1.77 pp | 43.08 % | +2.39 pp | 42.60 % | +1.91 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 17.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 61.72 % | 33.64 % | -28.08 pp | 33.11 % | -28.61 pp | 51.78 % | -9.94 pp |
| IPC Active | 1.53 inst/cycle | 2.67 inst/cycle | +1.14 | 2.64 inst/cycle | +1.11 | 1.93 inst/cycle | +0.40 |

---

## Cumulative Progress

```
Baseline:  21660 ns  (1.000×)

Final best: 21660 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
