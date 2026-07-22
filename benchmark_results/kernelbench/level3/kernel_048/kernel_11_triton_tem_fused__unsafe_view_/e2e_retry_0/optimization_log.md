# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 5980 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_11_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 121
- **Duration**: 5980 ns
- **SM Throughput**: 0.22 % | **Memory**: 0.62 % | **DRAM**: 0.22 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 1.53 inst/cycle | **No Eligible**: 58.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_and_fix_grid | 0.845× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5980 ns | 6080 ns | +100 ns (+1.7%) |
| SM Throughput (%) | 0.22 % | 0.22 % | +0.00 pp |
| Memory Throughput (%) | 0.62 % | 0.44 % | -0.18 pp |
| DRAM Throughput (%) | 0.22 % | 0.24 % | +0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 121.00 register/thread | 121.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 58.42 % | 61.56 % | +3.14 pp |
| IPC Active | 1.53 inst/cycle | 1.51 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)

Final best: 5980 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
