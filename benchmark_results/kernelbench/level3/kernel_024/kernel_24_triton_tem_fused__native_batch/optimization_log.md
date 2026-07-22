# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 57060 ns — **Best latency**: 27330 ns (**2.0878×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_block_m_and_block_k_to_match_problem_dimensions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_24_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9,1,1), Waves/SM: 0.02
- **Registers/thread**: 62
- **Duration**: 57060 ns
- **SM Throughput**: 0.54 % | **Memory**: 6.29 % | **DRAM**: 0.48 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.709× | 1.790× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy | 2.679× | 2.943× | ✓ |  |
| C | split_k_matmul_parallelism | 1.230× | 0.927× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_small_gemm | 2.895× | 3.218× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | split_k_matmul_pa... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57060 ns | 31870 ns | -25190 ns (-44.1%) | 19390 ns | -37670 ns (-66.0%) | 61570 ns | +4510 ns (+7.9%) | 17730 ns | -39330 ns (-68.9%) |
| SM Throughput (%) | 0.54 % | 0.24 % | -0.30 pp | 0.39 % | -0.15 pp | 0.48 % | -0.06 pp | 0.44 % | -0.10 pp |
| Memory Throughput (%) | 6.29 % | 0.86 % | -5.43 pp | 4.72 % | -1.57 pp | 5.83 % | -0.46 pp | 5.14 % | -1.15 pp |
| DRAM Throughput (%) | 0.48 % | 0.86 % | +0.38 pp | 1.42 % | +0.94 pp | 0.45 % | -0.03 pp | 1.55 % | +1.07 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 62.00 register/thread | 158.00 register/thread | +96 | 32.00 register/thread | -30 | 56.00 register/thread | -6 | 56.00 register/thread | -6 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 91.86 % | 84.94 % | -6.92 pp | 89.69 % | -2.17 pp | 92.64 % | +0.78 pp | 86.80 % | -5.06 pp |
| IPC Active | 0.33 inst/cycle | 0.15 inst/cycle | -0.18 | 0.20 inst/cycle | -0.13 | 0.29 inst/cycle | -0.04 | 0.26 inst/cycle | -0.07 |

---

## Iteration 2

**Best so far**: 3.2183× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_dot_with_register_mac_to_eliminate_smem_bank_conflicts | 1.029× | 1.068× | ✓ |  |
| B | reduce_block_k_to_eliminate_shared_memory_bank_conflicts | 2.929× | 2.024× | ✓ |  |
| C | reduce_block_k_to_cut_register_pressure_and_improve_occupancy | 1.047× | 1.092× | ✓ |  |
| D | reduce_block_m_and_block_k_to_match_problem_dimensions | 2.953× | 2.088× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_tl_dot_wi... | Δ | reduce_block_k_to... | Δ | reduce_block_k_to... | Δ | reduce_block_m_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 57060 ns | 53410 ns | -3650 ns (-6.4%) | 28190 ns | -28870 ns (-50.6%) | 52260 ns | -4800 ns (-8.4%) | 27330 ns | -29730 ns (-52.1%) |
| SM Throughput (%) | 0.54 % | 0.15 % | -0.39 pp | 0.21 % | -0.33 pp | 0.15 % | -0.39 pp | 0.21 % | -0.33 pp |
| Memory Throughput (%) | 6.29 % | 0.54 % | -5.75 pp | 1.73 % | -4.56 pp | 0.53 % | -5.76 pp | 1.73 % | -4.56 pp |
| DRAM Throughput (%) | 0.48 % | 0.51 % | +0.03 pp | 0.97 % | +0.49 pp | 0.52 % | +0.04 pp | 1.00 % | +0.52 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 62.00 register/thread | 80.00 register/thread | +18 | 64.00 register/thread | +2 | 80.00 register/thread | +18 | 64.00 register/thread | +2 |
| Waves/SM | 0.02 | 0.00 | -0.02 | 0.00 | -0.02 | 0.00 | -0.02 | 0.00 | -0.02 |
| No Eligible (%) | 91.86 % | 90.72 % | -1.14 pp | 88.04 % | -3.82 pp | 90.33 % | -1.53 pp | 88.00 % | -3.86 pp |
| IPC Active | 0.33 inst/cycle | 0.09 inst/cycle | -0.24 | 0.12 inst/cycle | -0.21 | 0.10 inst/cycle | -0.23 | 0.12 inst/cycle | -0.21 |


### Key Learnings

#### ✅ Plan B: `reduce_block_k_to_eliminate_shared_memory_bank_conflicts` (2.024×)

When a Triton matmul kernel exhibits severe shared memory bank conflicts (SharedMemoryConflicts rule, >70% excessive wavefronts) and CPIStall attributes >60% of stall cycles to MIO/shared-memory scoreboard waits, halving BLOCK_K reduces the shared memory tile footprint per block (17 KB → 8.7 KB), lowers average bank conflict way (17 → 12), and cuts MIO stall contribution (62.8% → 41.3%). Combined with an expanded autotune search over smaller num_warps values (1, 2) and deeper num_stages (4, 5) to compensate for the doubled k-loop iterations, this yields a 2.02x wall-time reduction. The strategy is most effective when Block Limit Shared Mem is the binding occupancy constraint and K is evenly divisible by the new BLOCK_K.

> **Applicable when**: Triton matmul kernel where: (1) SharedMemoryConflicts rule fires with >70% excessive wavefronts and >10-way average bank conflict; (2) CPIStall attributes >50% of stall cycles to MIO/shared-memory; (3) Block Limit Shared Mem is the binding occupancy constraint; (4) K is evenly divisible by the reduced BLOCK_K; (5) waves_per_sm < 0.1 (grid not SM-saturated).

> **Avoid when**: K is not evenly divisible by the new BLOCK_K (requires boundary masking overhead); kernel is compute-bound (Compute SM Throughput > 50%); shared memory is not the binding block limit; BLOCK_K is already at minimum warp-tile granularity (e.g., 32); reducing BLOCK_K would cause too many k-loop iterations to hide with available num_stages.

---

## Iteration 3

**Best so far**: 3.2183× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_block_k_to_restore_memory_throughput_and_reduce_loop_iterations | 2.659× | 2.899× | ✓ |  |
| C | restore_block_k_to_64_in_elementwise_mac_kernel | 2.659× | 1.919× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | restore_block_k_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 57060 ns | 19680 ns | -37380 ns (-65.5%) | 29730 ns | -27330 ns (-47.9%) |
| SM Throughput (%) | 0.54 % | 0.45 % | -0.09 pp | 0.32 % | -0.22 pp |
| Memory Throughput (%) | 6.29 % | 1.39 % | -4.90 pp | 0.92 % | -5.37 pp |
| DRAM Throughput (%) | 0.48 % | 1.39 % | +0.91 pp | 0.92 % | +0.44 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 62.00 register/thread | 149.00 register/thread | +87 | 255.00 register/thread | +193 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 91.86 % | 87.09 % | -4.77 pp | 83.39 % | -8.47 pp |
| IPC Active | 0.33 inst/cycle | 0.26 inst/cycle | -0.07 | 0.17 inst/cycle | -0.16 |

---

## Iteration 4

**Best so far**: 3.2183× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_sm_saturation | 2.671× | 2.835× | ✓ |  |
| D | persistent_kernel_one_block_per_sm_tile_loop | 2.636× | 1.932× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 57060 ns | 20130 ns | -36930 ns (-64.7%) | 29540 ns | -27520 ns (-48.2%) |
| SM Throughput (%) | 0.54 % | 0.35 % | -0.19 pp | 0.33 % | -0.21 pp |
| Memory Throughput (%) | 6.29 % | 1.36 % | -4.93 pp | 0.93 % | -5.36 pp |
| DRAM Throughput (%) | 0.48 % | 1.36 % | +0.88 pp | 0.93 % | +0.45 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 62.00 register/thread | 255.00 register/thread | +193 | 255.00 register/thread | +193 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 91.86 % | 79.23 % | -12.63 pp | 83.11 % | -8.75 pp |
| IPC Active | 0.33 inst/cycle | 0.20 inst/cycle | -0.13 | 0.17 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  57060 ns  (1.000×)
Iter 1     17730 ns  (3.218×)  — autotune_tile_sizes_for_underutilized_small_gemm

Final best: 27330 ns  (2.0878×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,15 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=2, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=2, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=3),
+    triton.Config({}, num_warps=8, num_stages=3),
+    triton.Config({}, num_warps=2, num_stages=4),
+    triton.Config({}, num_warps=4, num_stages=4),
+    triton.Config({}, num_warps=8, num_stages=4),
+    triton.Config({}, num_warps=4, num_stages=1),
+    triton.Config({}, num_warps=8, num_stages=1),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
-    GROUP_M : tl.constexpr = 8
+    BLOCK_K : tl.constexpr = 64
+    GROUP_M : tl.constexpr = 1
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
@@ -72,9 +85,8 @@

         xindex = idx_n + 288*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # Use 3D broadcast MAC to avoid TF32 precision issues while staying efficient
+        acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +100,6 @@

     tmp0 = tl.full([1], 0, tl.int32)
     tmp1 = tl.maximum(tmp0, acc)
     tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
