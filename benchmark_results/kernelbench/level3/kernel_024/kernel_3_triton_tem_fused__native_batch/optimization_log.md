# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 52380 ns — **Best latency**: 52380 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_3_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 3.71
- **Registers/thread**: 214
- **Duration**: 52380 ns
- **SM Throughput**: 23.73 % | **Memory**: 59.99 % | **DRAM**: 8.85 %
- **Achieved Occupancy**: 12.40 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 0.994× | ✓ |  |
| B | tma_loads_with_block_ptr_and_register_pressure_reduction | 0.987× | 0.984× | ✓ |  |
| C | eviction_policies_for_streaming_vs_reused_data | 0.993× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_bl... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 52380 ns | 52700 ns | +320 ns (+0.6%) | 53220 ns | +840 ns (+1.6%) | 52770 ns | +390 ns (+0.7%) |
| SM Throughput (%) | 23.73 % | 23.69 % | -0.04 pp | 23.55 % | -0.18 pp | 23.59 % | -0.14 pp |
| Memory Throughput (%) | 59.99 % | 59.90 % | -0.09 pp | 59.18 % | -0.81 pp | 59.42 % | -0.57 pp |
| DRAM Throughput (%) | 8.85 % | 8.74 % | -0.11 pp | 8.66 % | -0.19 pp | 8.55 % | -0.30 pp |
| Achieved Occupancy (%) | 12.40 % | 12.40 % | +0.00 pp | 12.33 % | -0.07 pp | 12.40 % | +0.00 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 246.00 register/thread | +32 | 214.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 72.96 % | 72.93 % | -0.03 pp | 73.17 % | +0.21 pp | 72.94 % | -0.02 pp |
| IPC Active | 1.08 inst/cycle | 1.08 inst/cycle | +0.00 | 1.07 inst/cycle | -0.01 | 1.08 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  52380 ns  (1.000×)

Final best: 52380 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
