# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_83` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10530 ns — **Best latency**: 10530 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_83`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_83_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12740,1,1), Waves/SM: 6.03
- **Registers/thread**: 22
- **Duration**: 10530 ns
- **SM Throughput**: 52.40 % | **Memory**: 20.02 % | **DRAM**: 12.69 %
- **Achieved Occupancy**: 61.42 %
- **IPC Active**: 2.65 inst/cycle | **No Eligible**: 32.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | — | fail | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_batchnorm_relu | — | fail | ✓ |  |
| C | vectorized_loads_stores_with_larger_xblock | — | fail | ✓ |  |
| D | autotune_xblock_and_num_warps_to_hide_l1tex_latency | — | fail | ✓ |  |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  10530 ns  (1.000×)

Final best: 10530 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
