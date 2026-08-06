# Kernel Optimization Log

## Summary

- **Kernel**: `partial_sum_sq_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1880000 ns — **Best latency**: 1880000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `partial_sum_sq_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_037/synthesis/kernel_0_fused_fused_linalg.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (7168,1,1), Waves/SM: 6.79
- **Registers/thread**: 19
- **Duration**: 1880000 ns
- **SM Throughput**: 52.37 % | **Memory**: 81.53 % | **DRAM**: 81.53 %
- **Achieved Occupancy**: 97.05 %
- **IPC Active**: 1.59 inst/cycle | **No Eligible**: 60.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_evict_first_expanded_autotune | 1.000× | 0.964× | ✓ |  |
| C | expand_autotune_with_vectorization_hints_and_eviction_policy | 0.999× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | expand_autotune_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1880000 ns | 1950000 ns | +70000 ns (+3.7%) | 1940000 ns | +60000 ns (+3.2%) |
| SM Throughput (%) | 52.37 % | 46.18 % | -6.19 pp | 45.94 % | -6.43 pp |
| Memory Throughput (%) | 81.53 % | 78.38 % | -3.15 pp | 78.77 % | -2.76 pp |
| DRAM Throughput (%) | 81.53 % | 78.38 % | -3.15 pp | 78.77 % | -2.76 pp |
| Achieved Occupancy (%) | 97.05 % | 88.82 % | -8.23 pp | 88.62 % | -8.43 pp |
| Registers/Thread | 19.00 register/thread | 19.00 register/thread | +0 | 19.00 register/thread | +0 |
| Waves/SM | 6.79 | 3.39 | -3.40 | 3.39 | -3.40 |
| No Eligible (%) | 60.77 % | 63.79 % | +3.02 pp | 64.73 % | +3.96 pp |
| IPC Active | 1.59 inst/cycle | 1.42 inst/cycle | -0.17 | 1.41 inst/cycle | -0.18 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  1880000 ns  (1.000×)

Final best: 1880000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
