# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244260 ns — **Best latency**: 244260 ns (**1.0000×** vs baseline)
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
- **Duration**: 244260 ns
- **SM Throughput**: 35.00 % | **Memory**: 32.01 % | **DRAM**: 32.01 %
- **Achieved Occupancy**: 43.81 %
- **IPC Active**: 1.42 inst/cycle | **No Eligible**: 64.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_for_latency_hiding | 0.996× | 0.792× | ✓ |  |
| B | increase_xblock_via_autotune_for_better_cache_reuse_and_parallelism | 0.876× | 0.891× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | increase_xblock_v... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 244260 ns | 308580 ns | +64320 ns (+26.3%) | 274140 ns | +29880 ns (+12.2%) |
| SM Throughput (%) | 35.00 % | 68.32 % | +33.32 pp | 38.30 % | +3.30 pp |
| Memory Throughput (%) | 32.01 % | 25.09 % | -6.92 pp | 28.23 % | -3.78 pp |
| DRAM Throughput (%) | 32.01 % | 25.09 % | -6.92 pp | 28.23 % | -3.78 pp |
| Achieved Occupancy (%) | 43.81 % | 68.84 % | +25.03 pp | 35.29 % | -8.52 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 372.36 | +186.18 | 186.18 | +0.00 |
| No Eligible (%) | 64.65 % | 32.08 % | -32.57 pp | 60.00 % | -4.65 pp |
| IPC Active | 1.42 inst/cycle | 2.71 inst/cycle | +1.29 | 1.57 inst/cycle | +0.15 |

---

## Cumulative Progress

```
Baseline:  244260 ns  (1.000×)

Final best: 244260 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
