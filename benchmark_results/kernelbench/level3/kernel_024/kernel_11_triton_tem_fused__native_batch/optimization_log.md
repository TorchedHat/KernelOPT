# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8190 ns — **Best latency**: 5860 ns (**1.3976×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_m_to_actual_m_and_shrink_block_n_to_relieve_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_11_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (18,1,1), Waves/SM: 0.01
- **Registers/thread**: 44
- **Duration**: 8190 ns
- **SM Throughput**: 0.92 % | **Memory**: 7.36 % | **DRAM**: 0.62 %
- **Achieved Occupancy**: 5.73 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_block_m_and_add_autotune_to_fill_grid | 0.684× | 0.586× | ✓ |  |
| C | split_k_to_increase_parallelism | 1.107× | 0.862× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_block_m_an... | Δ | split_k_to_increa... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 13980 ns | +5790 ns (+70.7%) | 9500 ns | +1310 ns (+16.0%) |
| SM Throughput (%) | 0.92 % | 1.81 % | +0.89 pp | 0.83 % | -0.09 pp |
| Memory Throughput (%) | 7.36 % | 8.90 % | +1.54 pp | 6.22 % | -1.14 pp |
| DRAM Throughput (%) | 0.62 % | 0.38 % | -0.24 pp | 0.52 % | -0.10 pp |
| Achieved Occupancy (%) | 5.73 % | 6.19 % | +0.46 pp | 5.79 % | +0.06 pp |
| Registers/Thread | 44.00 register/thread | 96.00 register/thread | +52 | 48.00 register/thread | +4 |
| Waves/SM | 0.01 | 0.03 | +0.02 | 0.01 | +0.00 |
| No Eligible (%) | 90.45 % | 84.21 % | -6.24 pp | 91.82 % | +1.37 pp |
| IPC Active | 0.34 inst/cycle | 0.63 inst/cycle | +0.29 | 0.31 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_dims_to_fix_grid_underutilization | 1.380× | 1.168× | ✓ | **✓ best** |
| B | persistent_kernel_scheduling | 1.331× | 1.462× | ✓ |  |
| C | tf32_tensor_cores | 0.677× | 0.588× | ✓ |  |
| D | persistent_kernel_one_block_per_sm | 0.805× | 0.637× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_dim... | Δ | persistent_kernel... | Δ | tf32_tensor_cores | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 7010 ns | -1180 ns (-14.4%) | 5600 ns | -2590 ns (-31.6%) | 13920 ns | +5730 ns (+70.0%) | 12860 ns | +4670 ns (+57.0%) |
| SM Throughput (%) | 0.92 % | 0.62 % | -0.30 pp | 0.60 % | -0.32 pp | 1.82 % | +0.90 pp | 0.44 % | -0.48 pp |
| Memory Throughput (%) | 7.36 % | 4.43 % | -2.93 pp | 5.57 % | -1.79 pp | 8.96 % | +1.60 pp | 2.64 % | -4.72 pp |
| DRAM Throughput (%) | 0.62 % | 0.71 % | +0.09 pp | 0.88 % | +0.26 pp | 0.38 % | -0.24 pp | 0.40 % | -0.22 pp |
| Achieved Occupancy (%) | 5.73 % | 6.24 % | +0.51 pp | 6.23 % | +0.50 pp | 6.20 % | +0.47 pp | 6.00 % | +0.27 pp |
| Registers/Thread | 44.00 register/thread | 34.00 register/thread | -10 | 32.00 register/thread | -12 | 96.00 register/thread | +52 | 93.00 register/thread | +49 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.03 | +0.02 | 0.03 | +0.02 |
| No Eligible (%) | 90.45 % | 93.45 % | +3.00 pp | 94.46 % | +4.01 pp | 84.11 % | -6.34 pp | 87.10 % | -3.35 pp |
| IPC Active | 0.34 inst/cycle | 0.26 inst/cycle | -0.08 | 0.22 inst/cycle | -0.12 | 0.63 inst/cycle | +0.29 | 0.49 inst/cycle | +0.15 |

---

## Iteration 3

**Best so far**: 1.1683× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_smem_padding | 1.408× | 1.185× | ✓ |  |
| B | eliminate_shared_memory_bank_conflicts_via_block_k_reduction | 0.717× | 0.522× | ✓ |  |
| C | reduce_shared_memory_bank_conflicts_via_blockk_reshape | 1.343× | 1.489× | ✓ |  |
| D | reduce_block_m_to_actual_m_and_shrink_block_n_to_relieve_register_pressure | 1.491× | 1.398× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | eliminate_shared_... | Δ | reduce_shared_mem... | Δ | reduce_block_m_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 6910 ns | -1280 ns (-15.6%) | 15680 ns | +7490 ns (+91.5%) | 5500 ns | -2690 ns (-32.8%) | 5860 ns | -2330 ns (-28.4%) |
| SM Throughput (%) | 0.92 % | 0.57 % | -0.35 pp | 1.42 % | +0.50 pp | 0.70 % | -0.22 pp | 0.37 % | -0.55 pp |
| Memory Throughput (%) | 7.36 % | 4.60 % | -2.76 pp | 4.93 % | -2.43 pp | 3.20 % | -4.16 pp | 2.61 % | -4.75 pp |
| DRAM Throughput (%) | 0.62 % | 0.71 % | +0.09 pp | 0.35 % | -0.27 pp | 0.90 % | +0.28 pp | 0.85 % | +0.23 pp |
| Achieved Occupancy (%) | 5.73 % | 6.24 % | +0.51 pp | 3.12 % | -2.61 pp | 6.23 % | +0.50 pp | 3.12 % | -2.61 pp |
| Registers/Thread | 44.00 register/thread | 90.00 register/thread | +46 | 164.00 register/thread | +120 | 34.00 register/thread | -10 | 34.00 register/thread | -10 |
| Waves/SM | 0.01 | 0.03 | +0.02 | 0.02 | +0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 90.45 % | 93.74 % | +3.29 pp | 75.91 % | -14.54 pp | 92.34 % | +1.89 pp | 92.16 % | +1.71 pp |
| IPC Active | 0.34 inst/cycle | 0.26 inst/cycle | -0.08 | 0.47 inst/cycle | +0.13 | 0.32 inst/cycle | -0.02 | 0.16 inst/cycle | -0.18 |

---

## Iteration 4

**Best so far**: 1.3976× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_num_warps_and_num_stages_to_reduce_smem_bank_conflicts | 1.402× | 1.089× | ✓ |  |
| B | reduce_block_k_to_cut_smem_bank_conflicts | 1.486× | 1.208× | ✓ |  |
| C | reduce_block_k_to_cut_smem_bank_conflicts | 1.418× | 1.108× | ✓ |  |
| D | reduce_block_m_to_actual_m_and_shrink_block_n_to_restore_occupancy | 1.471× | 1.108× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_num_warp... | Δ | reduce_block_k_to... | Δ | reduce_block_k_to... | Δ | reduce_block_m_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 7520 ns | -670 ns (-8.2%) | 6780 ns | -1410 ns (-17.2%) | 7390 ns | -800 ns (-9.8%) | 7390 ns | -800 ns (-9.8%) |
| SM Throughput (%) | 0.92 % | 0.22 % | -0.70 pp | 0.31 % | -0.61 pp | 0.56 % | -0.36 pp | 0.24 % | -0.68 pp |
| Memory Throughput (%) | 7.36 % | 1.01 % | -6.35 pp | 1.36 % | -6.00 pp | 2.54 % | -4.82 pp | 1.50 % | -5.86 pp |
| DRAM Throughput (%) | 0.62 % | 0.65 % | +0.03 pp | 0.73 % | +0.11 pp | 0.66 % | +0.04 pp | 0.66 % | +0.04 pp |
| Achieved Occupancy (%) | 5.73 % | 1.56 % | -4.17 pp | 3.12 % | -2.61 pp | 6.24 % | +0.51 pp | 3.12 % | -2.61 pp |
| Registers/Thread | 44.00 register/thread | 60.00 register/thread | +16 | 36.00 register/thread | -8 | 38.00 register/thread | -6 | 36.00 register/thread | -8 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 90.45 % | 91.50 % | +1.05 pp | 93.86 % | +3.41 pp | 93.44 % | +2.99 pp | 95.32 % | +4.87 pp |
| IPC Active | 0.34 inst/cycle | 0.08 inst/cycle | -0.26 | 0.12 inst/cycle | -0.22 | 0.25 inst/cycle | -0.09 | 0.10 inst/cycle | -0.24 |


### Key Learnings

#### ✅ Plan B: `reduce_block_k_to_cut_smem_bank_conflicts` (1.489×)

When the B matrix stride in the K dimension is a multiple of the number of shared memory banks (e.g., stride_bk=1 but stride_bn=96, and 96 mod 32 = 0), loading a [BLOCK_K × BLOCK_N] tile through shared memory causes all threads in a warp to map to the same bank, producing high-way bank conflicts (observed 17-way average, 82% of wavefronts). Halving BLOCK_K from 32 to 16 reduces the number of threads simultaneously issuing conflicting loads per wavefront, cutting average bank conflict ways from 17 to 9 and total bank conflicts from 48384 to 20736. This also halved dynamic shared memory per block (8.7→4.35 KB) and reduced L1/TEX cache throughput from 61.7% to 38.6%, with Warp Cycles Per Issued Instruction dropping from 17.6 to 12.7, yielding a 1.49x speedup despite a modest increase in instruction count (more K-loop iterations).

> **Applicable when**: GEMM kernel where the B matrix stride in the K dimension is a multiple of the shared memory bank count (stride mod 32 == 0 for 32-bank GPUs), causing high-way shared memory bank conflicts (NCU SharedMemoryConflicts rule triggered, bank_conflict_ways >= 8, bank_conflict_wavefronts_pct >= 60%). CPIStall shows MIO (shared memory) stalls as a dominant stall reason (>30% of cycles). L1/TEX Cache Throughput is elevated (>40%) relative to actual compute throughput.

> **Avoid when**: BLOCK_K is already small (<=8) and further reduction would cause tl.dot to fall back to scalar FFMA (losing MMA efficiency). Do not apply when the K dimension is not divisible by the new BLOCK_K (would require masking and break EVEN_K=True). Do not apply when the bank conflict root cause is in the A matrix access pattern rather than B. Also note: this increases the number of K-loop iterations (more loop overhead), which may dominate for very small K values.

#### ✅ Plan C: `reduce_block_k_to_cut_smem_bank_conflicts` (1.489×)

When the B matrix stride in the K dimension is a multiple of the number of shared memory banks (e.g., stride_bk=1 but stride_bn=96, and 96 mod 32 = 0), loading a [BLOCK_K × BLOCK_N] tile through shared memory causes all threads in a warp to map to the same bank, producing high-way bank conflicts (observed 17-way average, 82% of wavefronts). Halving BLOCK_K from 32 to 16 reduces the number of threads simultaneously issuing conflicting loads per wavefront, cutting average bank conflict ways from 17 to 9 and total bank conflicts from 48384 to 20736. This also halved dynamic shared memory per block (8.7→4.35 KB) and reduced L1/TEX cache throughput from 61.7% to 38.6%, with Warp Cycles Per Issued Instruction dropping from 17.6 to 12.7, yielding a 1.49x speedup despite a modest increase in instruction count (more K-loop iterations).

> **Applicable when**: GEMM kernel where the B matrix stride in the K dimension is a multiple of the shared memory bank count (stride mod 32 == 0 for 32-bank GPUs), causing high-way shared memory bank conflicts (NCU SharedMemoryConflicts rule triggered, bank_conflict_ways >= 8, bank_conflict_wavefronts_pct >= 60%). CPIStall shows MIO (shared memory) stalls as a dominant stall reason (>30% of cycles). L1/TEX Cache Throughput is elevated (>40%) relative to actual compute throughput.

> **Avoid when**: BLOCK_K is already small (<=8) and further reduction would cause tl.dot to fall back to scalar FFMA (losing MMA efficiency). Do not apply when the K dimension is not divisible by the new BLOCK_K (would require masking and break EVEN_K=True). Do not apply when the bank conflict root cause is in the A matrix access pattern rather than B. Also note: this increases the number of K-loop iterations (more loop overhead), which may dominate for very small K values.

---

## Iteration 5

**Best so far**: 1.3976× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_block_n_to_64_with_num_warps_1_in_autotune | 1.442× | 0.598× | ✓ |  |
| B | align_block_n_to_grid_size_via_autotune | 1.453× | 1.260× | ✓ |  |
| C | increase_block_n_to_reduce_loop_overhead_and_improve_instruction_efficiency | 1.458× | 1.104× | ✓ |  |
| D | remove_unnecessary_boundary_checks_and_enable_fast_accum | 1.405× | 1.137× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_block_n_to... | Δ | align_block_n_to_... | Δ | increase_block_n_... | Δ | remove_unnecessar... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8190 ns | 13700 ns | +5510 ns (+67.3%) | 6500 ns | -1690 ns (-20.6%) | 7420 ns | -770 ns (-9.4%) | 7200 ns | -990 ns (-12.1%) |
| SM Throughput (%) | 0.92 % | 0.22 % | -0.70 pp | 0.30 % | -0.62 pp | 0.43 % | -0.49 pp | 0.20 % | -0.72 pp |
| Memory Throughput (%) | 7.36 % | 3.35 % | -4.01 pp | 1.31 % | -6.05 pp | 1.98 % | -5.38 pp | 0.98 % | -6.38 pp |
| DRAM Throughput (%) | 0.62 % | 0.37 % | -0.25 pp | 0.75 % | +0.13 pp | 0.66 % | +0.04 pp | 0.68 % | +0.06 pp |
| Achieved Occupancy (%) | 5.73 % | 1.56 % | -4.17 pp | 3.12 % | -2.61 pp | 3.12 % | -2.61 pp | 1.56 % | -4.17 pp |
| Registers/Thread | 44.00 register/thread | 66.00 register/thread | +22 | 38.00 register/thread | -6 | 44.00 register/thread | +0 | 60.00 register/thread | +16 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 90.45 % | 92.21 % | +1.76 pp | 93.68 % | +3.23 pp | 90.92 % | +0.47 pp | 92.21 % | +1.76 pp |
| IPC Active | 0.34 inst/cycle | 0.08 inst/cycle | -0.26 | 0.13 inst/cycle | -0.21 | 0.17 inst/cycle | -0.17 | 0.08 inst/cycle | -0.26 |


### Key Learnings

#### ❌ Plan A: `expand_block_n_to_64_with_num_warps_1_in_autotune` (0.598×)

Changing BLOCK_N from 32 to 64 (while keeping BLOCK_M=2 and BLOCK_K=16) caused registers/thread to nearly double from 36 to 66, dropping theoretical occupancy from 75% to 43.75% and achieved occupancy from 3.12% to 1.56%. Critically, shared memory bank conflicts worsened dramatically: average conflict ways jumped from 9 to 24.7 and the fraction of conflicted wavefronts rose from 66.67% to 94.12%, because the wider BLOCK_N=64 tile generates more simultaneous conflicting accesses per wavefront for the stride-96 B matrix. The autotuner selected block_size=32 (num_warps=1), which halved the thread count per block (64→32 threads), further reducing parallelism. Despite Warp Cycles Per Issued Instruction improving slightly (16.91→12.98), the net effect was a 1.67x slowdown (6.78→13.7 µs) because register pressure and bank conflict degradation dominated.

> **Applicable when**: Never: this is a negative experience. Do not increase BLOCK_N from 32 to 64 for this kernel configuration (BLOCK_M=2, BLOCK_K=16, stride_bn=96). The register explosion and bank conflict amplification dominate any latency-hiding benefit.

> **Avoid when**: Do not increase BLOCK_N when: (1) the B matrix stride is a multiple of 32 (stride_bn mod 32 == 0), as wider tiles amplify bank conflicts rather than reducing them; (2) the kernel is already near the register threshold where a doubling of BLOCK_N pushes registers/thread above ~50, cutting theoretical occupancy below 50%; (3) adding num_warps=1 to the autotune search space risks the autotuner selecting it, which halves thread count per block and further reduces parallelism for an already grid-underutilized kernel.

---

## Cumulative Progress

```
Baseline:  8190 ns  (1.000×)
Iter 2     7010 ns  (1.168×)  — autotune_tile_dims_to_fix_grid_underutilization
Iter 3     5860 ns  (1.398×)  — reduce_block_m_to_actual_m_and_shrink_block_n_to_relieve_register_pressure

Final best: 5860 ns  (1.3976×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_11(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -104,8 +116,6 @@

     tmp16 = tl.maximum(tmp15, tmp14)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 576*idx_m, [BLOCK_M, BLOCK_N])), tmp16, mask)
 
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
