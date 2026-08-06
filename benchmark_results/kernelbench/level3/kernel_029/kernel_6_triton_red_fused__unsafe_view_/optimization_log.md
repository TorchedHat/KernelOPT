# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1230000 ns — **Best latency**: 1230000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_6_triton_red_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 1230000 ns
- **SM Throughput**: 11.44 % | **Memory**: 34.47 % | **DRAM**: 34.47 %
- **Achieved Occupancy**: 11.59 %
- **IPC Active**: 0.21 inst/cycle | **No Eligible**: 94.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints | 1.000× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1230000 ns | 1220000 ns | -10000 ns (-0.8%) |
| SM Throughput (%) | 11.44 % | 11.60 % | +0.16 pp |
| Memory Throughput (%) | 34.47 % | 34.62 % | +0.15 pp |
| DRAM Throughput (%) | 34.47 % | 34.62 % | +0.15 pp |
| Achieved Occupancy (%) | 11.59 % | 11.58 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 |
| No Eligible (%) | 94.79 % | 94.77 % | -0.02 pp |
| IPC Active | 0.21 inst/cycle | 0.21 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1230000 ns  (1.000×)

Final best: 1230000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
