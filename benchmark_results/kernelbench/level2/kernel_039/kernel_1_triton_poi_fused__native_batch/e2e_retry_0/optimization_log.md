# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 43100 ns — **Best latency**: 43100 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_039/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 43100 ns
- **SM Throughput**: 62.07 % | **Memory**: 31.35 % | **DRAM**: 19.42 %
- **Achieved Occupancy**: 57.83 %
- **IPC Active**: 2.62 inst/cycle | **No Eligible**: 34.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_num_warps_num_stages | 0.617× | 0.601× | ✓ |  |
| C | add_triton_autotune | 0.997× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_n... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 43100 ns | 71710 ns | +28610 ns (+66.4%) | 43300 ns | +200 ns (+0.5%) |
| SM Throughput (%) | 62.07 % | 83.21 % | +21.14 pp | 75.89 % | +13.82 pp |
| Memory Throughput (%) | 31.35 % | 28.54 % | -2.81 pp | 30.99 % | -0.36 pp |
| DRAM Throughput (%) | 19.42 % | 11.59 % | -7.83 pp | 19.27 % | -0.15 pp |
| Achieved Occupancy (%) | 57.83 % | 89.01 % | +31.18 pp | 69.81 % | +11.98 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 34.97 % | 14.38 % | -20.59 pp | 19.82 % | -15.15 pp |
| IPC Active | 2.62 inst/cycle | 3.42 inst/cycle | +0.80 | 3.20 inst/cycle | +0.58 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_num_warps_num_stages` (0.601×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) to a pointwise kernel that had low achieved occupancy (57.8%) and high CPIStall (46.9% on L1TEX) caused a 40% regression (0.60x). Although occupancy improved from 57.8% to 89.0% and eligible warps per scheduler rose from 1.37 to 7.46, the total executed instruction count more than doubled (20.9M → 46.9M) and registers per thread jumped from 22 to 32, reducing the block limit from 21 to 16 blocks/SM. The net effect was a 66% increase in kernel duration (43.1 µs → 71.7 µs). The autotune overhead combined with the larger register footprint and instruction explosion from the selected config outweighed the occupancy gain, indicating that for this kernel the original XBLOCK=128 with 22 registers was already near-optimal in terms of instruction efficiency, and the low occupancy was not the true bottleneck.

> **Applicable when**: Never apply this specific pattern (autotune XBLOCK upward + add xmask) when: registers per thread are already low (≤24), the kernel is purely elementwise with no shared memory, and the instruction count is already minimal. The occupancy gap (theoretical vs achieved) may be misleading if the kernel is already compute-efficient at the current tile size.

> **Avoid when**: Do not increase XBLOCK via autotune on pointwise kernels where: (1) registers per thread will increase significantly with larger blocks (e.g. 22→32), causing block limit reduction that offsets occupancy gains; (2) the selected autotune config causes instruction count to more than double; (3) the kernel already has low register pressure and the CPIStall is due to memory latency that larger tiles cannot hide without shared memory. Also avoid adding xmask when xnumel is a power-of-2 multiple of all candidate XBLOCK values — the mask adds unnecessary predication overhead.

---

## Cumulative Progress

```
Baseline:  43100 ns  (1.000×)

Final best: 43100 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
