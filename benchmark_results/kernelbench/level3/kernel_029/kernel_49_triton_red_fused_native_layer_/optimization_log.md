# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_native_layer_norm_view_49` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2490000 ns — **Best latency**: 2490000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_native_layer_norm_view_49`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_49_triton_red_fused_native_layer_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 2490000 ns
- **SM Throughput**: 0.32 % | **Memory**: 0.94 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.17 inst/cycle | **No Eligible**: 95.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorize_loads_via_full_r0_block_elimination_of_reduction_loop | 0.896× | 0.899× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorize_loads_v... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2490000 ns | 2770000 ns | +280000 ns (+11.2%) |
| SM Throughput (%) | 0.32 % | 0.34 % | +0.02 pp |
| Memory Throughput (%) | 0.94 % | 1.03 % | +0.09 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 95.85 % | 95.75 % | -0.10 pp |
| IPC Active | 0.17 inst/cycle | 0.17 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  2490000 ns  (1.000×)

Final best: 2490000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
