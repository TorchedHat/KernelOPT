# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244380 ns — **Best latency**: 244380 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_2_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244380 ns
- **SM Throughput**: 32.55 % | **Memory**: 48.17 % | **DRAM**: 48.17 %
- **Achieved Occupancy**: 45.84 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 85.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_plus_xblock_increase | 0.738× | 0.754× | ✓ |  |
| B | vectorized_loads_with_contiguity_hints | 0.752× | 0.691× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 244380 ns | 324290 ns | +79910 ns (+32.7%) | 353540 ns | +109160 ns (+44.7%) |
| SM Throughput (%) | 32.55 % | 24.87 % | -7.68 pp | 46.60 % | +14.05 pp |
| Memory Throughput (%) | 48.17 % | 36.26 % | -11.91 pp | 33.27 % | -14.90 pp |
| DRAM Throughput (%) | 48.17 % | 36.26 % | -11.91 pp | 33.27 % | -14.90 pp |
| Achieved Occupancy (%) | 45.84 % | 43.85 % | -1.99 pp | 43.21 % | -2.63 pp |
| Registers/Thread | 16.00 register/thread | 22.00 register/thread | +6 | 22.00 register/thread | +6 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 372.36 | +186.18 |
| No Eligible (%) | 85.98 % | 80.21 % | -5.77 pp | 73.03 % | -12.95 pp |
| IPC Active | 0.56 inst/cycle | 0.77 inst/cycle | +0.21 | 1.10 inst/cycle | +0.54 |


### Key Learnings

#### ❌ Plan A: `eviction_policy_streaming_plus_xblock_increase` (0.754×)

Adding eviction_policy='evict_first' and @triton.autotune with larger XBLOCK configs to a streaming pointwise kernel caused a 25% slowdown. Despite reducing Warp Cycles Per Issued Instruction from 50.9 to 35.1, the total instruction count nearly doubled (26.7M → 44.8M) and DRAM throughput dropped from 48% to 36%, indicating the autotune overhead and/or the larger XBLOCK introduced more instructions (boundary masking, register pressure from 16→22 registers/thread) without proportionally improving memory-level parallelism. The net effect was a longer kernel duration (244 µs → 324 µs) despite the per-instruction stall improvement.

> **Applicable when**: Never — this specific combination caused regression. evict_first alone may be worth testing separately. Autotune with larger XBLOCK may be worth testing separately with correct grid lambda.

> **Avoid when**: Streaming pointwise kernels where the grid is not updated to use meta['XBLOCK'] when autotune changes XBLOCK — the grid remained hardcoded to cdiv(xnumel, 128) while autotune selected larger XBLOCK values, causing the grid to over-launch and each CTA to process more elements than intended, inflating instruction count. Also avoid when register pressure increase (16→22 regs/thread) would reduce occupancy on already occupancy-limited kernels.

---

## Cumulative Progress

```
Baseline:  244380 ns  (1.000×)

Final best: 244380 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
