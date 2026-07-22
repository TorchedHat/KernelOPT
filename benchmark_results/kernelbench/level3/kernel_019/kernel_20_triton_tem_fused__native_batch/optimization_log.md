# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 112670 ns — **Best latency**: 112670 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_20_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.24
- **Registers/thread**: 128
- **Duration**: 112670 ns
- **SM Throughput**: 21.65 % | **Memory**: 85.01 % | **DRAM**: 1.13 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 76.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.864× | 0.869× | ✓ |  |
| B | autotune_tile_sizes_with_tma_block_ptr_loads | 0.428× | 0.419× | ✓ |  |
| D | add_autotune_increase_tile_sizes_fix_bank_conflicts | 0.996× | 0.576× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 112670 ns | 129600 ns | +16930 ns (+15.0%) | 268860 ns | +156190 ns (+138.6%) | 195740 ns | +83070 ns (+73.7%) |
| SM Throughput (%) | 21.65 % | 18.80 % | -2.85 pp | 35.52 % | +13.87 pp | 13.17 % | -8.48 pp |
| Memory Throughput (%) | 85.01 % | 74.32 % | -10.69 pp | 81.73 % | -3.28 pp | 92.74 % | +7.73 pp |
| DRAM Throughput (%) | 1.13 % | 0.98 % | -0.15 pp | 0.48 % | -0.65 pp | 0.65 % | -0.48 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.50 % | +6.25 pp | 12.50 % | +6.25 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 255.00 register/thread | +127 | 80.00 register/thread | -48 |
| Waves/SM | 0.24 | 0.32 | +0.08 | 0.97 | +0.73 | 0.32 | +0.08 |
| No Eligible (%) | 76.88 % | 79.62 % | +2.74 pp | 62.91 % | -13.97 pp | 86.28 % | +9.40 pp |
| IPC Active | 0.92 inst/cycle | 0.81 inst/cycle | -0.11 | 1.49 inst/cycle | +0.57 | 0.55 inst/cycle | -0.37 |


### Key Learnings

#### ❌ Plan A: `tf32_tensor_cores` (0.869×)

Increasing BLOCK_K from 32 to 64 doubled the dynamic shared memory per block (32.77 KB → 65.54 KB), which reduced the shared-memory block limit from 4 to 3 blocks per SM and dropped theoretical occupancy from 25% to 18.75%. Despite the intent to enable TF32 Tensor Cores, the diff shows ALLOW_TF32 was never changed from False, so no Tensor Core benefit was realized. The net effect was more shared memory pressure, lower occupancy, higher warp stall cycles (4.33 → 4.91 cycles/instruction), and a 15% longer kernel duration. When a kernel is already bottlenecked by shared memory bank conflicts (77% excessive wavefronts) and low occupancy, increasing BLOCK_K worsens the shared memory footprint and further constrains occupancy without delivering any compute throughput improvement.

> **Applicable when**: Never beneficial as applied here. The BLOCK_K increase without the corresponding ALLOW_TF32=True flip is a pure regression.

> **Avoid when**: Do NOT increase BLOCK_K when: (1) the kernel already has severe shared memory bank conflicts (>70% excessive wavefronts), (2) shared memory is already a block-limit bottleneck (Block Limit Shared Mem ≤ 4), (3) the ALLOW_TF32 flag was not actually changed (implementation flaw). Increasing tile size in a shared-memory-bound kernel with bank conflicts amplifies the bottleneck rather than relieving it.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_eliminate_shared_memory_bank_conflicts | 0.868× | 0.869× | ✓ |  |
| B | fix_b_matrix_shared_memory_bank_conflicts_via_transposed_load | 0.998× | 0.575× | ✓ |  |
| C | reduce_tile_sizes_to_eliminate_register_spill | 0.940× | 0.941× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_shar... | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 112670 ns | 129660 ns | +16990 ns (+15.1%) | 195810 ns | +83140 ns (+73.8%) | 119780 ns | +7110 ns (+6.3%) |
| SM Throughput (%) | 21.65 % | 18.81 % | -2.84 pp | 13.17 % | -8.48 pp | 20.86 % | -0.79 pp |
| Memory Throughput (%) | 85.01 % | 74.34 % | -10.67 pp | 92.75 % | +7.74 pp | 80.46 % | -4.55 pp |
| DRAM Throughput (%) | 1.13 % | 0.98 % | -0.15 pp | 0.65 % | -0.48 pp | 1.06 % | -0.07 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.50 % | +6.25 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 80.00 register/thread | -48 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.32 | +0.08 | 0.32 | +0.08 | 0.24 | +0.00 |
| No Eligible (%) | 76.88 % | 79.66 % | +2.78 pp | 86.28 % | +9.40 pp | 78.08 % | +1.20 pp |
| IPC Active | 0.92 inst/cycle | 0.81 inst/cycle | -0.11 | 0.55 inst/cycle | -0.37 | 0.88 inst/cycle | -0.04 |


### Key Learnings

#### ❌ Plan A: `fix_b_matrix_load_eliminate_shared_memory_bank_conflicts` (0.869×)

Replacing a double tl.broadcast_to B-matrix load with a clean strided pointer expression (b_k_idx_vals * stride_bk + offs_b_n[None, :] * stride_bn) produced no measurable change in any profiling metric: bank conflict rate remained 12-way average, excessive wavefronts stayed at 77.72%, Warp Cycles Per Issued Instruction held at ~4.91, and duration was unchanged at ~129.6 µs. This indicates the shared memory bank conflicts are not caused by the Python-level broadcast expression in the Triton source — Triton's compiler generates the same shared memory staging layout regardless of whether the index is expressed as a double broadcast or a clean strided product. The root cause of the bank conflicts lies deeper in the kernel's shared memory layout (tile shape, padding, or access stride relative to the 32-bank structure) and cannot be fixed by reformulating the pointer arithmetic alone.

> **Applicable when**: Never applicable as a standalone fix for shared memory bank conflicts in Triton GEMM kernels. Reformulating the pointer arithmetic expression does not change the shared memory staging layout.

> **Avoid when**: When the kernel shows SharedMemoryConflicts (12-way average, >70% excessive wavefronts) and UncoalescedSharedAccess rules triggered: do NOT attempt to fix bank conflicts by reformulating the B-matrix load pointer expression from double-broadcast to strided arithmetic. Triton generates the same shared memory layout regardless of the Python-level index expression. The fix must address the shared memory tile layout itself (e.g., adding padding to the shared memory allocation, changing tile dimensions to avoid bank-stride alignment, or restructuring the access pattern at the tl.dot level).

#### ❌ Plan B: `fix_b_matrix_shared_memory_bank_conflicts_via_transposed_load` (0.575×)

Loading the B matrix as [BLOCK_N, BLOCK_K] and calling tl.trans() before tl.dot() was expected to eliminate shared memory bank conflicts by changing the in-register layout, but the bank conflict rate actually worsened (12-way → 17-way, 77.72% → 82.32% excessive wavefronts) and Warp Cycles Per Issued Instruction tripled (4.91 → 14.57). The profiling data shows that the confounding factor was a simultaneous block_size increase from 128 to 256 (introduced via @triton.autotune), which amplified the conflicts as documented in prior iterations — the transposed load alone was never isolated. The net result was a 51% longer kernel duration, with L1/TEX Cache Throughput rising to 96.6% and Mem Busy to 92.75%, indicating the shared memory bottleneck intensified rather than resolved.

> **Applicable when**: Never demonstrated to work in isolation for this kernel. The transposed load strategy may be valid in principle but has not been tested without the confounding block_size increase.

> **Avoid when**: Do NOT combine a B-matrix transposed load change with a block_size increase via @triton.autotune when the kernel already has severe shared memory bank conflicts (>70% excessive wavefronts). The block_size increase dominates and amplifies conflicts regardless of the load pattern change. If testing transposed B load, hold block_size constant at the original value (128) to isolate the effect. Also do not apply when the autotune config space includes larger block sizes that will be selected over the original.

---

## Cumulative Progress

```
Baseline:  112670 ns  (1.000×)

Final best: 112670 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_20(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -87,8 +88,6 @@

     xindex = idx_n + 1024*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
