# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 38080 ns — **Best latency**: 38080 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_31_triton_tem_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (176,1,1), Waves/SM: 0.44
- **Registers/thread**: 168
- **Duration**: 38080 ns
- **SM Throughput**: 22.97 % | **Memory**: 51.68 % | **DRAM**: 0.66 %
- **Achieved Occupancy**: 8.95 %
- **IPC Active**: 1.28 inst/cycle | **No Eligible**: 67.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 1.009× | ✓ |  |
| C | annotate_streaming_loads_with_evict_first_policy | 0.995× | 1.005× | ✓ |  |
| D | add_autotune_reduce_register_pressure_via_smaller_tiles | 0.991× | 0.698× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | annotate_streamin... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38080 ns | 37730 ns | -350 ns (-0.9%) | 37890 ns | -190 ns (-0.5%) | 54560 ns | +16480 ns (+43.3%) |
| SM Throughput (%) | 22.97 % | 22.85 % | -0.12 pp | 22.85 % | -0.12 pp | 15.72 % | -7.25 pp |
| Memory Throughput (%) | 51.68 % | 51.41 % | -0.27 pp | 51.37 % | -0.31 pp | 25.99 % | -25.69 pp |
| DRAM Throughput (%) | 0.66 % | 0.66 % | +0.00 pp | 0.66 % | +0.00 pp | 0.47 % | -0.19 pp |
| Achieved Occupancy (%) | 8.95 % | 8.95 % | +0.00 pp | 8.96 % | +0.01 pp | 4.19 % | -4.76 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.44 | 0.44 | +0.00 | 0.44 | +0.00 | 0.33 | -0.11 |
| No Eligible (%) | 67.65 % | 67.67 % | +0.02 pp | 67.72 % | +0.07 pp | 72.85 % | +5.20 pp |
| IPC Active | 1.28 inst/cycle | 1.29 inst/cycle | +0.01 | 1.29 inst/cycle | +0.01 | 0.73 inst/cycle | -0.55 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | cap_registers_with_maxnreg_to_improve_occupancy | 0.918× | 0.854× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 38080 ns | 44580 ns | +6500 ns (+17.1%) |
| SM Throughput (%) | 22.97 % | 21.09 % | -1.88 pp |
| Memory Throughput (%) | 51.68 % | 47.44 % | -4.24 pp |
| DRAM Throughput (%) | 0.66 % | 0.58 % | -0.08 pp |
| Achieved Occupancy (%) | 8.95 % | 8.98 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 128.00 register/thread | -40 |
| Waves/SM | 0.44 | 0.33 | -0.11 |
| No Eligible (%) | 67.65 % | 70.47 % | +2.82 pp |
| IPC Active | 1.28 inst/cycle | 1.18 inst/cycle | -0.10 |


### Key Learnings

#### ❌ Plan B: `cap_registers_with_maxnreg_to_improve_occupancy` (0.854×)

Applying maxnreg via @triton.autotune to cap register allocation from 168 to 128 registers/thread successfully raised theoretical occupancy from 18.75% to 25% (Block Limit Registers: 3→4), but achieved occupancy remained nearly identical (8.95%→8.98%) and kernel duration increased from 37.73us to 44.58us (18% regression). The register cap triggered heavy register spilling to local memory, evidenced by new MemoryCacheAccessPattern warnings for local loads/stores with only 1.0 of 32 bytes utilized per sector—a classic spill pattern. The spill traffic increased total instructions by ~8% and introduced uncoalesced local memory accesses that more than offset any occupancy benefit. When the kernel is already wave-count-limited (0.44 waves/SM) and the dominant stall is shared memory bank conflicts (7.9-way, ~50% of stall cycles), capping registers to gain one extra block per SM does not help because the bottleneck is not latency-hiding capacity but rather the bank conflict serialization itself.

> **Applicable when**: Never applicable as a standalone fix when: (1) the kernel has a large accumulator tile (BLOCK_M x BLOCK_N >= 128x64) that already requires >100 registers/thread, AND (2) the dominant stall is shared memory bank conflicts rather than pure latency-hiding deficit. maxnreg may help in kernels where register pressure is only slightly above a block-count threshold and the compiler can spill cheaply without introducing uncoalesced local memory patterns.

> **Avoid when**: Kernel has a large GEMM accumulator tile (BLOCK_M*BLOCK_N >= 8192 elements) with registers_per_thread > 128. In this regime, capping maxnreg forces the compiler to spill the accumulator registers to local memory with stride-1-per-thread access patterns (1/32 bytes utilized per sector), creating severe uncoalesced local memory traffic that more than offsets any occupancy gain. The achieved occupancy will not improve meaningfully if the kernel is already wave-count-limited and the primary bottleneck is shared memory bank conflicts.

---

## Cumulative Progress

```
Baseline:  38080 ns  (1.000×)

Final best: 38080 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused_convolution_31(arg_A, arg_B, out_ptr0):
@@ -90,8 +91,6 @@

     xindex = idx_n + 672*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
