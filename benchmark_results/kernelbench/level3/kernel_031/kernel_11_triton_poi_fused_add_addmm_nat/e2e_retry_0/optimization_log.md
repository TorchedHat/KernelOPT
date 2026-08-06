# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_native_layer_norm_permute_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15420 ns — **Best latency**: 15420 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_native_layer_norm_permute_view_11`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_11_triton_poi_fused_add_addmm_nat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 3.45
- **Registers/thread**: 56
- **Duration**: 15420 ns
- **SM Throughput**: 39.12 % | **Memory**: 74.28 % | **DRAM**: 47.69 %
- **Achieved Occupancy**: 48.96 %
- **IPC Active**: 1.85 inst/cycle | **No Eligible**: 53.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_alignment_hints | 0.990× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 15420 ns | 15490 ns | +70 ns (+0.5%) |
| SM Throughput (%) | 39.12 % | 39.36 % | +0.24 pp |
| Memory Throughput (%) | 74.28 % | 74.74 % | +0.46 pp |
| DRAM Throughput (%) | 47.69 % | 47.87 % | +0.18 pp |
| Achieved Occupancy (%) | 48.96 % | 48.97 % | +0.01 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 |
| Waves/SM | 3.45 | 3.45 | +0.00 |
| No Eligible (%) | 53.72 % | 53.76 % | +0.04 pp |
| IPC Active | 1.85 inst/cycle | 1.85 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  15420 ns  (1.000×)

Final best: 15420 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
