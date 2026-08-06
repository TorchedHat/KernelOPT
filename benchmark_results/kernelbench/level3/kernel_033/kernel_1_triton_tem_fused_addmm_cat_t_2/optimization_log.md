# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_cat_t_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 14880000 ns — **Best latency**: 14880000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_cat_t_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_1_triton_tem_fused_addmm_cat_t_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 14880000 ns
- **SM Throughput**: 40.62 % | **Memory**: 58.34 % | **DRAM**: 3.21 %
- **Achieved Occupancy**: 12.23 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_for_streaming_k_loop_loads | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 14880000 ns | 14880000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.62 % | 40.61 % | -0.01 pp |
| Memory Throughput (%) | 58.34 % | 58.34 % | +0.00 pp |
| DRAM Throughput (%) | 3.21 % | 3.21 % | +0.00 pp |
| Achieved Occupancy (%) | 12.23 % | 12.23 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 58.34 % | 58.33 % | -0.01 pp |
| IPC Active | 1.67 inst/cycle | 1.67 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  14880000 ns  (1.000×)

Final best: 14880000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
