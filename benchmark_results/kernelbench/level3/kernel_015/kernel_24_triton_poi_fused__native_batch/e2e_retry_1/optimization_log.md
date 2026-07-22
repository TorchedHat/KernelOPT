# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5280 ns — **Best latency**: 5280 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_24_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 1.86
- **Registers/thread**: 18
- **Duration**: 5280 ns
- **SM Throughput**: 33.52 % | **Memory**: 12.27 % | **DRAM**: 7.80 %
- **Achieved Occupancy**: 62.87 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 38.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_via_autotune_to_eliminate_partial_wave | 1.000× | 0.363× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.999× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_v... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5280 ns | 14560 ns | +9280 ns (+175.8%) | 5250 ns | -30 ns (-0.6%) |
| SM Throughput (%) | 33.52 % | 75.37 % | +41.85 pp | 23.75 % | -9.77 pp |
| Memory Throughput (%) | 12.27 % | 15.81 % | +3.54 pp | 10.66 % | -1.61 pp |
| DRAM Throughput (%) | 7.80 % | 2.83 % | -4.97 pp | 7.84 % | +0.04 pp |
| Achieved Occupancy (%) | 62.87 % | 53.63 % | -9.24 pp | 66.21 % | +3.34 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 24.00 register/thread | +6 |
| Waves/SM | 1.86 | 5.94 | +4.08 | 0.93 | -0.93 |
| No Eligible (%) | 38.55 % | 11.41 % | -27.14 pp | 31.33 % | -7.22 pp |
| IPC Active | 2.43 inst/cycle | 3.54 inst/cycle | +1.11 | 2.28 inst/cycle | -0.15 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_via_autotune_to_eliminate_partial_wave` (0.363×)

Adding @triton.autotune with larger XBLOCK candidates (256–2048) to eliminate a partial-wave tail caused a severe regression: the selected config (XBLOCK=256) tripled register usage from 18 to 48 registers/thread, collapsing Block Limit Registers from 21 to 5 and theoretical occupancy from 100% to 62.5%. This forced 5.94 waves instead of the intended single wave, increasing executed instructions by 6.3x and duration from 5.28 µs to 14.56 µs. When a kernel already has low register pressure and high theoretical occupancy at the baseline block size, increasing XBLOCK via autotune can cause register spilling or register-limited occupancy collapse that far outweighs any partial-wave benefit.

> **Applicable when**: Kernel has high partial-wave overhead (LaunchConfiguration rule fires with >30% estimated speedup), low register pressure at baseline, and theoretical occupancy near 100% — meaning there is headroom to increase block size without register-limited occupancy collapse.

> **Avoid when**: Kernel already uses near-maximum registers per thread (>24 registers at block_size=128), or when increasing XBLOCK causes register count to jump significantly (e.g., 18→48), collapsing Block Limit Registers and theoretical occupancy. Also avoid when the call-site grid is not updated to use meta['XBLOCK'] from autotune (static grid with old block size causes incorrect work distribution).

---

## Cumulative Progress

```
Baseline:  5280 ns  (1.000×)

Final best: 5280 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
