# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244860 ns — **Best latency**: 244860 ns (**1.0000×** vs baseline)
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
- **Duration**: 244860 ns
- **SM Throughput**: 32.51 % | **Memory**: 48.07 % | **DRAM**: 48.07 %
- **Achieved Occupancy**: 45.59 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 85.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_stores_plus_larger_xblock | 0.999× | 1.002× | ✓ |  |
| B | vectorized_loads_with_autotune | 0.753× | 0.655× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 244860 ns | 244350 ns | -510 ns (-0.2%) | 373540 ns | +128680 ns (+52.6%) |
| SM Throughput (%) | 32.51 % | 32.74 % | +0.23 pp | 35.20 % | +2.69 pp |
| Memory Throughput (%) | 48.07 % | 48.09 % | +0.02 pp | 31.47 % | -16.60 pp |
| DRAM Throughput (%) | 48.07 % | 48.09 % | +0.02 pp | 31.47 % | -16.60 pp |
| Achieved Occupancy (%) | 45.59 % | 48.08 % | +2.49 pp | 43.67 % | -1.92 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 372.36 | +186.18 |
| No Eligible (%) | 85.98 % | 76.06 % | -9.92 pp | 65.77 % | -20.21 pp |
| IPC Active | 0.56 inst/cycle | 0.95 inst/cycle | +0.39 | 1.36 inst/cycle | +0.80 |


### Key Learnings

#### ❌ Plan B: `vectorized_loads_with_autotune` (0.655×)

Adding @triton.autotune with larger XBLOCK configs and tl.max_contiguous/tl.multiple_of alignment hints on a pointwise kernel caused a 52% slowdown. Although warp stall cycles per instruction improved (51.3→20.2) and eligible warps per scheduler increased (0.15→0.61), the total executed instruction count ballooned ~3.7x (26.7M→98.7M) and DRAM throughput dropped from 48% to 31%, resulting in a 53% longer kernel duration (245→374 us). The autotune selected block_size=256 (not the intended 512–2048), and the grid lambda was not updated to use meta['XBLOCK'], so the grid remained fixed at cdiv(xnumel, 128)=393216 blocks while the kernel now processed 256 elements per block — doubling total threads and work without any bandwidth benefit.

> **Applicable when**: Never — this is a negative example documenting an implementation flaw where the grid was not updated to match the autotuned XBLOCK.

> **Avoid when**: When adding @triton.autotune and removing the explicit XBLOCK argument from the launch call, always update the grid to use a lambda: `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. Failing to do so causes the grid to remain sized for the old (smaller) XBLOCK while the kernel processes a larger XBLOCK, resulting in redundant work, doubled thread count, and DRAM throughput regression.

---

## Cumulative Progress

```
Baseline:  244860 ns  (1.000×)

Final best: 244860 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
