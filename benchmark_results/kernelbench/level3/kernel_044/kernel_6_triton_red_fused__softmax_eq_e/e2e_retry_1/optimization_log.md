# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10070000 ns — **Best latency**: 10070000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_6_triton_red_fused__softmax_eq_e/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 32
- **Duration**: 10070000 ns
- **SM Throughput**: 14.02 % | **Memory**: 68.68 % | **DRAM**: 68.68 %
- **Achieved Occupancy**: 93.05 %
- **IPC Active**: 0.44 inst/cycle | **No Eligible**: 88.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_fix_xblock_r0block | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 10070000 ns | 10070000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 14.02 % | 14.04 % | +0.02 pp |
| Memory Throughput (%) | 68.68 % | 68.67 % | -0.01 pp |
| DRAM Throughput (%) | 68.68 % | 68.67 % | -0.01 pp |
| Achieved Occupancy (%) | 93.05 % | 92.96 % | -0.09 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 88.96 % | 88.93 % | -0.03 pp |
| IPC Active | 0.44 inst/cycle | 0.44 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  10070000 ns  (1.000×)

Final best: 10070000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
