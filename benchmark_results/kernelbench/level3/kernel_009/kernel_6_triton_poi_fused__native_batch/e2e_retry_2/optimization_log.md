# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4540 ns — **Best latency**: 4540 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 22
- **Duration**: 4540 ns
- **SM Throughput**: 26.44 % | **Memory**: 14.52 % | **DRAM**: 14.52 %
- **Achieved Occupancy**: 69.37 %
- **IPC Active**: 2.12 inst/cycle | **No Eligible**: 48.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave_tail | 0.942× | 0.826× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.943× | 0.830× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4540 ns | 5500 ns | +960 ns (+21.1%) | 5470 ns | +930 ns (+20.5%) |
| SM Throughput (%) | 26.44 % | 47.62 % | +21.18 pp | 48.91 % | +22.47 pp |
| Memory Throughput (%) | 14.52 % | 14.20 % | -0.32 pp | 14.57 % | +0.05 pp |
| DRAM Throughput (%) | 14.52 % | 11.96 % | -2.56 pp | 12.03 % | -2.49 pp |
| Achieved Occupancy (%) | 69.37 % | 75.37 % | +6.00 pp | 75.34 % | +5.97 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.48 | 1.48 | +0.00 | 1.48 | +0.00 |
| No Eligible (%) | 48.31 % | 21.32 % | -26.99 pp | 20.91 % | -27.40 pp |
| IPC Active | 2.12 inst/cycle | 3.11 inst/cycle | +0.99 | 3.11 inst/cycle | +0.99 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.830×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) to a short-duration pointwise kernel caused a 20% regression despite the LaunchConfiguration rule flagging 50% potential speedup from the partial wave. The autotune overhead itself (benchmark runs across configs) dominates for short kernels, and the selected larger XBLOCK increased register pressure from 22 to 32 registers/thread, inflating total executed instructions by 2.2x (965K → 2129K) without proportional throughput gains. The kernel duration increased from 4.54 µs to 5.47 µs, indicating that for very short-lived kernels (<10 µs), autotune overhead and register spill costs outweigh partial-wave savings.

> **Applicable when**: Kernel duration > 50 µs, partial wave accounts for >30% of runtime, register count is low (<20), and the workload is not latency-bound by a short kernel duration.

> **Avoid when**: Kernel duration is very short (<10 µs); autotune benchmark overhead dominates. Also avoid when the grid lambda is not updated to use meta['XBLOCK'] — the grid remained hardcoded to cdiv(xnumel, 128) in this implementation, meaning larger XBLOCK configs still ran 3136 blocks but with more work per block, inflating instruction count without reducing grid size. Additionally, adding xmask guards where the original used None (unconditional) adds unnecessary predication overhead for divisible problem sizes.

---

## Cumulative Progress

```
Baseline:  4540 ns  (1.000×)

Final best: 4540 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
