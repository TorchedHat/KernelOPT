# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45150 ns — **Best latency**: 45150 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_5_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 4.95
- **Registers/thread**: 168
- **Duration**: 45150 ns
- **SM Throughput**: 30.55 % | **Memory**: 77.12 % | **DRAM**: 13.28 %
- **Achieved Occupancy**: 17.94 %
- **IPC Active**: 1.33 inst/cycle | **No Eligible**: 66.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 0.991× | ✓ |  |
| B | reduce_block_n_to_eliminate_shared_memory_bank_conflicts | 0.981× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_block_n_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45150 ns | 45540 ns | +390 ns (+0.9%) | 46620 ns | +1470 ns (+3.3%) |
| SM Throughput (%) | 30.55 % | 30.42 % | -0.13 pp | 29.92 % | -0.63 pp |
| Memory Throughput (%) | 77.12 % | 76.78 % | -0.34 pp | 76.44 % | -0.68 pp |
| DRAM Throughput (%) | 13.28 % | 13.12 % | -0.16 pp | 12.79 % | -0.49 pp |
| Achieved Occupancy (%) | 17.94 % | 17.91 % | -0.03 pp | 17.91 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.76 % | 66.82 % | +0.06 pp | 67.57 % | +0.81 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.30 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  45150 ns  (1.000×)

Final best: 45150 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
