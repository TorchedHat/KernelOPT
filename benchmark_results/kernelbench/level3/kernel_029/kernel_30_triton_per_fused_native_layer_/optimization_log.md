# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_native_layer_norm_view_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 259840 ns — **Best latency**: 259840 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_native_layer_norm_view_30`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_30_triton_per_fused_native_layer_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.01
- **Registers/thread**: 32
- **Duration**: 259840 ns
- **SM Throughput**: 2.06 % | **Memory**: 5.92 % | **DRAM**: 1.29 %
- **Achieved Occupancy**: 6.17 %
- **IPC Active**: 0.32 inst/cycle | **No Eligible**: 91.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorize_loads_stores_via_r0block_power_of_two_alignment | 0.979× | 0.983× | ✓ |  |
| D | fix_grid_launch_xblock1_one_cta_per_row | 0.515× | 0.510× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorize_loads_s... | Δ | fix_grid_launch_x... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 259840 ns | 264380 ns | +4540 ns (+1.7%) | 509470 ns | +249630 ns (+96.1%) |
| SM Throughput (%) | 2.06 % | 2.32 % | +0.26 pp | 1.52 % | -0.54 pp |
| Memory Throughput (%) | 5.92 % | 6.51 % | +0.59 pp | 4.98 % | -0.94 pp |
| DRAM Throughput (%) | 1.29 % | 1.23 % | -0.06 pp | 2.35 % | +1.06 pp |
| Achieved Occupancy (%) | 6.17 % | 6.24 % | +0.07 pp | 6.02 % | -0.15 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.95 % | 91.71 % | -0.24 pp | 94.42 % | +2.47 pp |
| IPC Active | 0.32 inst/cycle | 0.33 inst/cycle | +0.01 | 0.21 inst/cycle | -0.11 |


### Key Learnings

#### ❌ Plan D: `fix_grid_launch_xblock1_one_cta_per_row` (0.510×)

The attempted fix of setting XBLOCK=1 (one CTA per row) to expand the grid from 16 to 1960 blocks did not improve performance — instead it caused a 2x regression (259us → 509us). Critically, the diff shows the actual code change did NOT modify XBLOCK or the grid size at all (grid_size and block_size remain 16 and 128 respectively in both before/after profiling). The only actual changes were cosmetic: adding explicit `mask=` keyword arguments to tl.load/tl.store calls. These mask keyword changes increased L2 input sectors from 2.1M to 3.4M, worsened CPIStall from 66% to 76.9%, increased warp cycles per instruction from 12.49 to 17.92, and added 32 instructions per scheduler — all consistent with the mask= keyword forcing less aggressive compiler optimizations or changing memory access patterns. The structural anti-pattern is: adding explicit `mask=` keyword to tl.load/tl.store when the mask was already being passed positionally can change compiler behavior and increase memory traffic.

> **Applicable when**: Never — this change is a regression. Do not add explicit mask= keyword to tl.load/tl.store calls that already pass mask positionally.

> **Avoid when**: Always avoid converting positional mask arguments to explicit mask= keyword arguments in tl.load/tl.store. The change increases L2 sector traffic (2.1M → 3.4M sectors), worsens CPIStall (66% → 76.9%), and increases warp cycles per instruction (12.49 → 17.92), resulting in ~2x wall-time regression. Additionally, note that the intended grid expansion (XBLOCK=1, grid=1960) was NOT actually implemented — the grid remained at 16 blocks — meaning the correctness failures seen in iterations 1 and 2 for this direction may be due to implementation errors at the call site rather than a fundamentally flawed strategy.

---

## Cumulative Progress

```
Baseline:  259840 ns  (1.000×)

Final best: 259840 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
