# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__softmax_add_clamp_clone_exp_index_log_mul_permute_prepare_softmax_online_sigmoid_sub_unsqueeze_view_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 40640 ns — **Best latency**: 40640 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__softmax_add_clamp_clone_exp_index_log_mul_permute_prepare_softmax_online_sigmoid_sub_unsqueeze_view_10`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_10_triton_per_fused__softmax_add_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (735,1,1), Waves/SM: 2.78
- **Registers/thread**: 255
- **Duration**: 40640 ns
- **SM Throughput**: 41.07 % | **Memory**: 40.30 % | **DRAM**: 13.49 %
- **Achieved Occupancy**: 11.62 %
- **IPC Active**: 1.85 inst/cycle | **No Eligible**: 53.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | online_softmax_attention | 0.368× | 0.375× | ✓ |  |
| D | vectorized_loads_stores_with_eviction_policy | 0.977× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | online_softmax_at... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 40640 ns | 108480 ns | +67840 ns (+166.9%) | 41730 ns | +1090 ns (+2.7%) |
| SM Throughput (%) | 41.07 % | 14.81 % | -26.26 pp | 41.11 % | +0.04 pp |
| Memory Throughput (%) | 40.30 % | 59.00 % | +18.70 pp | 39.19 % | -1.11 pp |
| DRAM Throughput (%) | 13.49 % | 6.45 % | -7.04 pp | 12.81 % | -0.68 pp |
| Achieved Occupancy (%) | 11.62 % | 21.76 % | +10.14 pp | 11.66 % | +0.04 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 |
| Waves/SM | 2.78 | 1.39 | -1.39 | 2.78 | +0.00 |
| No Eligible (%) | 53.93 % | 80.01 % | +26.08 pp | 53.99 % | +0.06 pp |
| IPC Active | 1.85 inst/cycle | 0.79 inst/cycle | -1.06 | 1.84 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  40640 ns  (1.000×)

Final best: 40640 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
