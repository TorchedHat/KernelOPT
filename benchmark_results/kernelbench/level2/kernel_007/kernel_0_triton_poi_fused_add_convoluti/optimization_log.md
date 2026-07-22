# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1140000 ns — **Best latency**: 1140000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_007/kernel_0_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1845120,1,1), Waves/SM: 873.64
- **Registers/thread**: 16
- **Duration**: 1140000 ns
- **SM Throughput**: 68.84 % | **Memory**: 33.30 % | **DRAM**: 33.30 %
- **Achieved Occupancy**: 63.19 %
- **IPC Active**: 2.76 inst/cycle | **No Eligible**: 31.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_hardcoded_no_autotune | 1.000× | 1.000× | ✓ |  |
| B | increase_xblock_static_for_latency_hiding_and_coalescing | 1.000× | 1.000× | ✓ |  |
| C | increase_xblock_direct_no_autotune_vectorized_loads | 0.999× | 1.000× | ✓ |  |
| D | manual_xblock_increase_to_1024_with_num_warps_8 | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_h... | Δ | increase_xblock_s... | Δ | increase_xblock_d... | Δ | manual_xblock_inc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1140000 ns | 1140000 ns | +0 ns (+0.0%) | 1140000 ns | +0 ns (+0.0%) | 1140000 ns | +0 ns (+0.0%) | 1140000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 68.84 % | 68.76 % | -0.08 pp | 68.84 % | +0.00 pp | 68.76 % | -0.08 pp | 68.76 % | -0.08 pp |
| Memory Throughput (%) | 33.30 % | 33.30 % | +0.00 pp | 33.30 % | +0.00 pp | 33.29 % | -0.01 pp | 33.31 % | +0.01 pp |
| DRAM Throughput (%) | 33.30 % | 33.30 % | +0.00 pp | 33.30 % | +0.00 pp | 33.29 % | -0.01 pp | 33.31 % | +0.01 pp |
| Achieved Occupancy (%) | 63.19 % | 63.28 % | +0.09 pp | 63.29 % | +0.10 pp | 63.24 % | +0.05 pp | 63.17 % | -0.02 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 873.64 | 873.64 | +0.00 | 873.64 | +0.00 | 873.64 | +0.00 | 873.64 | +0.00 |
| No Eligible (%) | 31.13 % | 31.03 % | -0.10 pp | 31.03 % | -0.10 pp | 31.09 % | -0.04 pp | 31.09 % | -0.04 pp |
| IPC Active | 2.76 inst/cycle | 2.76 inst/cycle | +0.00 | 2.76 inst/cycle | +0.00 | 2.76 inst/cycle | +0.00 | 2.76 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1140000 ns  (1.000×)

Final best: 1140000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
