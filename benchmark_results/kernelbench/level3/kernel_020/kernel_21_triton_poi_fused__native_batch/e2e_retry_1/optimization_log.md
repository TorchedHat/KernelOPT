# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3170 ns — **Best latency**: 3170 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 0.46
- **Registers/thread**: 18
- **Duration**: 3170 ns
- **SM Throughput**: 13.04 % | **Memory**: 4.82 % | **DRAM**: 3.30 %
- **Achieved Occupancy**: 35.62 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 64.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_xblock | 0.988× | 0.901× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 0.999× | 0.604× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3170 ns | 3520 ns | +350 ns (+11.0%) | 5250 ns | +2080 ns (+65.6%) |
| SM Throughput (%) | 13.04 % | 16.98 % | +3.94 pp | 49.30 % | +36.26 pp |
| Memory Throughput (%) | 4.82 % | 5.87 % | +1.05 pp | 10.57 % | +5.75 pp |
| DRAM Throughput (%) | 3.30 % | 2.98 % | -0.32 pp | 2.00 % | -1.30 pp |
| Achieved Occupancy (%) | 35.62 % | 32.59 % | -3.03 pp | 48.49 % | +12.87 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 48.00 register/thread | +30 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 1.48 | +1.02 |
| No Eligible (%) | 64.48 % | 55.12 % | -9.36 pp | 20.67 % | -43.81 pp |
| IPC Active | 1.32 inst/cycle | 1.77 inst/cycle | +0.45 | 3.02 inst/cycle | +1.70 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_increase_tile_size` (0.604×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) caused the autotuner to select XBLOCK=256 with 48 registers/thread (up from 18), reducing theoretical occupancy from 100% to 62.5% due to register pressure. Although waves_per_sm improved (0.46→1.48) and compute throughput rose (13%→49%), the increased instruction count (333K→2.08M executed instructions) and register-limited occupancy caused a net 1.66x slowdown. When a kernel's register count is already low at the baseline tile size, larger tiles can cause register spill or pressure that more than offsets the latency-hiding benefit.

> **Applicable when**: Kernel has very low waves_per_sm (<0.5) AND low register count per thread (<24) AND the bottleneck is confirmed as latency hiding (high No Eligible warp %, L1TEX stalls). The register budget must have headroom to absorb larger tiles without hitting the block-limit-registers constraint.

> **Avoid when**: The kernel already uses a moderate-to-high register count (>24 registers/thread) at the baseline tile size, because larger XBLOCK configs will cause the autotuner to select a config that increases register pressure, reducing theoretical occupancy below the baseline and causing net regression. Also avoid when the kernel's instruction count scales super-linearly with tile size.

---

## Cumulative Progress

```
Baseline:  3170 ns  (1.000×)

Final best: 3170 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
