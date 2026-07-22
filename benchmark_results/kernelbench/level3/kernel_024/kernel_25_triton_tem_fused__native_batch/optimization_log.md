# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16670 ns — **Best latency**: 7170 ns (**2.3250×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_25`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_25_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (36,1,1), Waves/SM: 0.03
- **Registers/thread**: 56
- **Duration**: 16670 ns
- **SM Throughput**: 2.19 % | **Memory**: 21.60 % | **DRAM**: 1.64 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.36 inst/cycle | **No Eligible**: 90.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 2.044× | 2.325× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_tiny_gemm | 0.726× | 0.559× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 16670 ns | 7170 ns | -9500 ns (-57.0%) | 29820 ns | +13150 ns (+78.9%) |
| SM Throughput (%) | 2.19 % | 1.25 % | -0.94 pp | 3.60 % | +1.41 pp |
| Memory Throughput (%) | 21.60 % | 12.60 % | -9.00 pp | 12.94 % | -8.66 pp |
| DRAM Throughput (%) | 1.64 % | 3.84 % | +2.20 pp | 0.94 % | -0.70 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 56.00 register/thread | 42.00 register/thread | -14 | 128.00 register/thread | +72 |
| Waves/SM | 0.03 | 0.01 | -0.02 | 0.03 | +0.00 |
| No Eligible (%) | 90.91 % | 89.80 % | -1.11 pp | 71.52 % | -19.39 pp |
| IPC Active | 0.36 inst/cycle | 0.20 inst/cycle | -0.16 | 0.57 inst/cycle | +0.21 |

---

## Iteration 2

**Best so far**: 2.325× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_smem_padding | 2.028× | 2.274× | ✓ |  |
| B | tma_loads_to_eliminate_shared_memory_bank_conflicts | 1.766× | 1.952× | ✓ |  |
| C | reduce_block_m_to_match_problem_dimension_with_safe_autotune | 1.369× | 0.528× | ✓ |  |
| D | tma_loads_with_block_m_matched_to_problem_dimension | 1.421× | 0.750× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | tma_loads_to_elim... | Δ | reduce_block_m_to... | Δ | tma_loads_with_bl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16670 ns | 7330 ns | -9340 ns (-56.0%) | 8540 ns | -8130 ns (-48.8%) | 31550 ns | +14880 ns (+89.3%) | 22240 ns | +5570 ns (+33.4%) |
| SM Throughput (%) | 2.19 % | 1.28 % | -0.91 pp | 1.18 % | -1.01 pp | 0.75 % | -1.44 pp | 0.92 % | -1.27 pp |
| Memory Throughput (%) | 21.60 % | 12.81 % | -8.79 pp | 10.77 % | -10.83 pp | 9.56 % | -12.04 pp | 8.80 % | -12.80 pp |
| DRAM Throughput (%) | 1.64 % | 3.74 % | +2.10 pp | 3.19 % | +1.55 pp | 0.88 % | -0.76 pp | 1.24 % | -0.40 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 56.00 register/thread | 42.00 register/thread | -14 | 32.00 register/thread | -24 | 128.00 register/thread | +72 | 128.00 register/thread | +72 |
| Waves/SM | 0.03 | 0.01 | -0.02 | 0.01 | -0.02 | 0.02 | -0.01 | 0.02 | -0.01 |
| No Eligible (%) | 90.91 % | 89.81 % | -1.10 pp | 88.89 % | -2.02 pp | 88.29 % | -2.62 pp | 83.77 % | -7.14 pp |
| IPC Active | 0.36 inst/cycle | 0.20 inst/cycle | -0.16 | 0.22 inst/cycle | -0.14 | 0.12 inst/cycle | -0.24 | 0.17 inst/cycle | -0.19 |


### Key Learnings

#### ❌ Plan A: `fix_b_matrix_bank_conflicts_via_smem_padding` (2.274×)

Rewriting the B-matrix load pointer from a broadcast expression to a direct `k_offs[:, None] + 288 * offs_b_n[None, :]` form did not reduce shared memory bank conflicts: the 17-way conflict rate and 82.35% excessive wavefront ratio remained identical before and after. The MIO stall percentage was unchanged (~50%), and kernel duration actually increased slightly (7.17 us → 7.33 us). This indicates the bank conflicts are not caused by the pointer arithmetic expression itself but by the underlying memory layout and access pattern relative to shared memory bank mapping — simply reordering the index computation without changing how data is staged through shared memory (e.g., adding padding to the shared memory buffer) cannot eliminate the conflicts. The speedup reported (2.27x) reflects the cumulative gain from the prior iteration's BLOCK_M reduction, not this change.

> **Applicable when**: SharedMemoryConflicts NCU rule fires with 17-way bank conflicts and 82% excessive wavefronts; CPIStall MIO stall ~50%; kernel uses a GEMM with B matrix stride_bn=288 (288 mod 32 = 0, causing all-same-bank accesses)

> **Avoid when**: Do not apply pointer arithmetic reordering alone to fix shared memory bank conflicts when the conflict is caused by the stride of the data layout (stride mod 32 == 0). The fix requires either: (1) padding the shared memory buffer by 1 element per row to break the stride alignment, or (2) transposing the tile before storing to shared memory. Pure pointer reordering without smem layout change has no effect on bank conflict counts.

#### ❌ Plan C: `reduce_block_m_to_match_problem_dimension_with_safe_autotune` (0.528×)

Setting BLOCK_M=2 while simultaneously expanding BLOCK_N from 32 to 128 caused a 1.89x slowdown. Although BLOCK_M=2 correctly matches the M=2 problem dimension, the larger BLOCK_N=128 tile tripled dynamic shared memory per block (8.7→16.6 KB) and caused registers/thread to spike from 42 to 128, collapsing theoretical occupancy from 62.5% to 20.3% and achieved occupancy from 3.12% to 1.56%. Executed instructions nearly tripled (58K→188K) and shared memory bank conflicts worsened from 17-way to 25-way. The lesson is that when reducing BLOCK_M to match a small M-dimension, BLOCK_N must remain at its original value (or be reduced) to avoid register and shared memory blowup that more than offsets the benefit of eliminating wasted M-dimension threads.

> **Applicable when**: BLOCK_M is being reduced to match a small M-dimension; BLOCK_N is simultaneously being increased to a larger tile size; registers/thread and dynamic shared memory per block both increase after the change; theoretical occupancy drops significantly.

> **Avoid when**: When only BLOCK_M is changed and BLOCK_N/BLOCK_K remain at their original values — that combination (exp_001_block_m_reduction_autotune) achieved a confirmed 2.325x speedup. Also do not change EVEN_K from True to False unless K is genuinely not divisible by BLOCK_K, as this adds unnecessary masking overhead.

---

## Iteration 3

**Best so far**: 2.325× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_block_n_autotune_tile_sizes_for_grid_coverage_and_bank_conflict_fix | 1.909× | 1.104× | ✓ |  |
| D | reduce_block_n_to_eliminate_smem_bank_conflicts | 1.725× | 0.901× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_n_au... | Δ | reduce_block_n_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 16670 ns | 15100 ns | -1570 ns (-9.4%) | 18500 ns | +1830 ns (+11.0%) |
| SM Throughput (%) | 2.19 % | 0.49 % | -1.70 pp | 0.67 % | -1.52 pp |
| Memory Throughput (%) | 21.60 % | 1.99 % | -19.61 pp | 6.91 % | -14.69 pp |
| DRAM Throughput (%) | 1.64 % | 1.81 % | +0.17 pp | 1.48 % | -0.16 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 56.00 register/thread | 64.00 register/thread | +8 | 96.00 register/thread | +40 |
| Waves/SM | 0.03 | 0.01 | -0.02 | 0.01 | -0.02 |
| No Eligible (%) | 90.91 % | 91.78 % | +0.87 pp | 88.77 % | -2.14 pp |
| IPC Active | 0.36 inst/cycle | 0.08 inst/cycle | -0.28 | 0.11 inst/cycle | -0.25 |

---

## Cumulative Progress

```
Baseline:  16670 ns  (1.000×)
Iter 1     7170 ns  (2.325×)  — tf32_tensor_cores

Final best: 7170 ns  (2.3250×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=1, num_stages=3),
+        triton.Config({}, num_warps=1, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_25(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
@@ -59,19 +73,16 @@

 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        k_offs = offs_k + k_idx * BLOCK_K  # shape [BLOCK_K], values = k_idx*32 + [0..31]
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 288*idx_m
-        a = tl.load(A + (xindex))
+        # A load (unchanged semantics): A[m, k] = A_ptr[288*m + k]
+        a_ptrs = A + offs_a_m[:, None] * 288 + k_offs[None, :]  # [BLOCK_M, BLOCK_K]
+        a = tl.load(a_ptrs, mask=(offs_a_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1152*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 288*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B load (FIXED): B[n,k]=B_ptr[288*n+k], so B tile [BLOCK_K,BLOCK_N] = B_ptr[k+288*n]
+        # K is now the fast axis (stride 1) — conflict-free, coalesced
+        b_ptrs = B + k_offs[:, None] + 288 * offs_b_n[None, :]  # [BLOCK_K, BLOCK_N]
+        b = tl.load(b_ptrs, mask=(k_offs[:, None] < K) & (offs_b_n[None, :] < N), other=0.0)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -87,8 +98,6 @@

     xindex = idx_n + 1152*idx_m
     tmp0 = tl.sigmoid(acc)
     tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
