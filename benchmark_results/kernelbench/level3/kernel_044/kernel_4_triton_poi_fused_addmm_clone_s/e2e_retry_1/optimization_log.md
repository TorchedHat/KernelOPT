# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 243970 ns — **Best latency**: 243970 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_4_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 243970 ns
- **SM Throughput**: 35.22 % | **Memory**: 32.04 % | **DRAM**: 32.04 %
- **Achieved Occupancy**: 43.79 %
- **IPC Active**: 1.42 inst/cycle | **No Eligible**: 64.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_persistent_tile_scheduling | 0.996× | 0.275× | ✓ |  |
| B | increase_xblock_tile_size_via_autotune | 0.876× | 0.885× | ✓ |  |
| C | add_triton_autotune | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243970 ns | 888100 ns | +644130 ns (+264.0%) | 275710 ns | +31740 ns (+13.0%) | 244130 ns | +160 ns (+0.1%) |
| SM Throughput (%) | 35.22 % | 77.12 % | +41.90 pp | 38.96 % | +3.74 pp | 37.43 % | +2.21 pp |
| Memory Throughput (%) | 32.04 % | 24.66 % | -7.38 pp | 28.07 % | -3.97 pp | 32.03 % | -0.01 pp |
| DRAM Throughput (%) | 32.04 % | 8.70 % | -23.34 pp | 28.07 % | -3.97 pp | 32.03 % | -0.01 pp |
| Achieved Occupancy (%) | 43.79 % | 75.79 % | +32.00 pp | 35.59 % | -8.20 pp | 44.40 % | +0.61 pp |
| Registers/Thread | 16.00 register/thread | 30.00 register/thread | +14 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 744.73 | +558.55 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 64.62 % | 22.93 % | -41.69 pp | 60.43 % | -4.19 pp | 61.90 % | -2.72 pp |
| IPC Active | 1.42 inst/cycle | 3.09 inst/cycle | +1.67 | 1.57 inst/cycle | +0.15 | 1.52 inst/cycle | +0.10 |

---

## Cumulative Progress

```
Baseline:  243970 ns  (1.000×)

Final best: 243970 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
