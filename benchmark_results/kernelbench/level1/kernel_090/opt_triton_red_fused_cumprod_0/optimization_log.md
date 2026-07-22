# Kernel Optimization Log

## Summary

- **Kernel**: `cumprod_kernel_v2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2190000 ns — **Best latency**: 2190000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `cumprod_kernel_v2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_090/synthesis/kernel_0_triton_red_fused_cumprod_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 20.69
- **Registers/thread**: 40
- **Duration**: 2190000 ns
- **SM Throughput**: 38.45 % | **Memory**: 79.65 % | **DRAM**: 79.65 %
- **Achieved Occupancy**: 74.18 %
- **IPC Active**: 1.54 inst/cycle | **No Eligible**: 61.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_alignment_hints | 0.987× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2190000 ns | 2180000 ns | -10000 ns (-0.5%) |
| SM Throughput (%) | 38.45 % | 38.43 % | -0.02 pp |
| Memory Throughput (%) | 79.65 % | 80.00 % | +0.35 pp |
| DRAM Throughput (%) | 79.65 % | 80.00 % | +0.35 pp |
| Achieved Occupancy (%) | 74.18 % | 74.19 % | +0.01 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 |
| Waves/SM | 20.69 | 20.69 | +0.00 |
| No Eligible (%) | 61.57 % | 61.51 % | -0.06 pp |
| IPC Active | 1.54 inst/cycle | 1.54 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  2190000 ns  (1.000×)

Final best: 2190000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
