# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11550 ns — **Best latency**: 11140 ns (**1.0368×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_stages_to_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_16`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_16_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (27,1,1), Waves/SM: 0.02
- **Registers/thread**: 56
- **Duration**: 11550 ns
- **SM Throughput**: 1.64 % | **Memory**: 13.34 % | **DRAM**: 0.93 %
- **Achieved Occupancy**: 5.89 %
- **IPC Active**: 0.37 inst/cycle | **No Eligible**: 89.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | split_k_matmul_parallelism | 1.197× | 0.536× | ✓ |  |
| D | increase_tile_sizes_and_autotune_for_small_gemm_epilogue | 1.282× | 1.114× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | split_k_matmul_pa... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11550 ns | 21540 ns | +9990 ns (+86.5%) | 10370 ns | -1180 ns (-10.2%) |
| SM Throughput (%) | 1.64 % | 0.64 % | -1.00 pp | 1.52 % | -0.12 pp |
| Memory Throughput (%) | 13.34 % | 4.01 % | -9.33 pp | 8.05 % | -5.29 pp |
| DRAM Throughput (%) | 0.93 % | 0.51 % | -0.42 pp | 1.03 % | +0.10 pp |
| Achieved Occupancy (%) | 5.89 % | 1.56 % | -4.33 pp | 6.25 % | +0.36 pp |
| Registers/Thread | 56.00 register/thread | 96.00 register/thread | +40 | 54.00 register/thread | -2 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.02 | +0.00 |
| No Eligible (%) | 89.96 % | 85.89 % | -4.07 pp | 90.35 % | +0.39 pp |
| IPC Active | 0.37 inst/cycle | 0.14 inst/cycle | -0.23 | 0.37 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.1138× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_to_reduce_shared_memory_bank_conflicts | 0.994× | 0.732× | ✓ |  |
| C | match_block_m_to_actual_m_dim_and_widen_block_n | 1.073× | 0.952× | ✓ |  |
| D | match_block_m_to_actual_m_dim_fix_register_pressure | 1.265× | 1.337× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | match_block_m_to_... | Δ | match_block_m_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11550 ns | 15780 ns | +4230 ns (+36.6%) | 12130 ns | +580 ns (+5.0%) | 8640 ns | -2910 ns (-25.2%) |
| SM Throughput (%) | 1.64 % | 1.55 % | -0.09 pp | 1.28 % | -0.36 pp | 2.09 % | +0.45 pp |
| Memory Throughput (%) | 13.34 % | 9.59 % | -3.75 pp | 8.44 % | -4.90 pp | 10.04 % | -3.30 pp |
| DRAM Throughput (%) | 0.93 % | 0.69 % | -0.24 pp | 0.88 % | -0.05 pp | 1.25 % | +0.32 pp |
| Achieved Occupancy (%) | 5.89 % | 6.25 % | +0.36 pp | 6.25 % | +0.36 pp | 6.25 % | +0.36 pp |
| Registers/Thread | 56.00 register/thread | 106.00 register/thread | +50 | 56.00 register/thread | +0 | 56.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.07 | +0.05 | 0.02 | +0.00 | 0.03 | +0.01 |
| No Eligible (%) | 89.96 % | 91.11 % | +1.15 pp | 92.31 % | +2.35 pp | 86.16 % | -3.80 pp |
| IPC Active | 0.37 inst/cycle | 0.36 inst/cycle | -0.01 | 0.31 inst/cycle | -0.06 | 0.55 inst/cycle | +0.18 |

---

## Iteration 3

**Best so far**: 1.1138× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_bank_conflicts_via_smem_padding | 1.087× | 0.967× | ✓ |  |
| B | reduce_block_k_to_eliminate_masked_loop_and_smem_bank_conflicts | 1.382× | 1.516× | ✓ | **✓ best** |
| C | reduce_shared_memory_bank_conflicts_via_padding | 1.289× | 1.104× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | reduce_block_k_to... | Δ | reduce_shared_mem... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11550 ns | 11940 ns | +390 ns (+3.4%) | 7620 ns | -3930 ns (-34.0%) | 10460 ns | -1090 ns (-9.4%) |
| SM Throughput (%) | 1.64 % | 1.27 % | -0.37 pp | 1.60 % | -0.04 pp | 1.44 % | -0.20 pp |
| Memory Throughput (%) | 13.34 % | 8.37 % | -4.97 pp | 5.93 % | -7.41 pp | 8.57 % | -4.77 pp |
| DRAM Throughput (%) | 0.93 % | 0.90 % | -0.03 pp | 1.42 % | +0.49 pp | 1.02 % | +0.09 pp |
| Achieved Occupancy (%) | 5.89 % | 6.25 % | +0.36 pp | 6.24 % | +0.35 pp | 6.24 % | +0.35 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 41.00 register/thread | -15 | 48.00 register/thread | -8 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 89.96 % | 92.24 % | +2.28 pp | 88.33 % | -1.63 pp | 89.47 % | -0.49 pp |
| IPC Active | 0.37 inst/cycle | 0.30 inst/cycle | -0.07 | 0.45 inst/cycle | +0.08 | 0.42 inst/cycle | +0.05 |


### Key Learnings

#### ✅ Plan B: `reduce_block_k_to_eliminate_masked_loop_and_smem_bank_conflicts` (1.516×)

When BLOCK_K does not evenly divide K, the final K-loop iteration processes mostly masked (invalid) elements, wasting compute and memory bandwidth. Choosing a BLOCK_K that exactly divides K (EVEN_K=True) eliminates mask computation and all partial-tile overhead. Simultaneously, reducing BLOCK_K shrinks the shared memory tile used by tl.dot, which lowers the shared memory footprint per block (33 KB → 17 KB), increases Block Limit Shared Mem (6 → 11), raises theoretical occupancy (37.5% → 62.5%), and reduces shared memory bank conflict severity (11.8-way → 6.6-way conflicts, MIO stall cycles 4.0 → 2.7). The combined effect is fewer executed instructions (143K → 93K), lower register pressure (56 → 41 regs/thread), and a 1.52x end-to-end speedup.

> **Applicable when**: SharedMemoryConflicts NCU rule fires with high bank-conflict-way (>8) and conflict wavefront % (>60%); BLOCK_K does not evenly divide K (EVEN_K=False); shared memory per block is large enough to limit Block Limit Shared Mem to a low value (<8); a smaller BLOCK_K value exists that exactly divides K; kernel is a GEMM or GEMM-like reduction loop in Triton.

> **Avoid when**: BLOCK_K already evenly divides K (EVEN_K already True) so no masking overhead exists; reducing BLOCK_K would increase the number of K-loop iterations to the point where loop overhead dominates; the kernel is already compute-bound on tensor cores and reducing tile size would hurt utilization; BLOCK_K is already at the minimum (e.g. 16) and further reduction is not possible.

#### ✅ Plan C: `reduce_shared_memory_bank_conflicts_via_padding` (1.104×)

Replacing raw pointer tl.load with tl.make_block_ptr (order=(1,0)) and expanding @triton.autotune configs to include num_stages=2,3,4 achieved only a modest 1.10x speedup despite the kernel having an 11.8-way shared memory bank conflict (77.67% of wavefronts). Critically, the bank conflict metrics were completely unchanged after the rewrite (still 11.8-way, 77.67%), meaning tl.make_block_ptr did NOT eliminate the bank conflicts as intended. The speedup came entirely from reduced register pressure (56→48 regs/thread) and halved dynamic shared memory (33→16 KB/block), which raised theoretical occupancy from 37.5% to 62.5%. However, elapsed cycles actually increased (12892→15656), and warp cycles per issued instruction worsened (7.48→9.48), indicating the autotune selected a config with more pipeline stages that added overhead without resolving the underlying bank conflict.

> **Applicable when**: SharedMemoryConflicts NCU rule fires with high bank conflict way (>8x) and high wavefront percentage (>70%); kernel uses raw pointer tl.load for GEMM tiles; the goal is to reduce register pressure as a side effect (registers per thread drops when manual masking variables are removed).

> **Avoid when**: The bank conflict is caused by a column-major B matrix stride that is a multiple of 32 (e.g. stride_bk=1, stride_bn=144 where 864%32=0); in this case tl.make_block_ptr with order=(1,0) does NOT eliminate the bank conflict — the 11.8-way conflict persists unchanged. Do not expect this approach to fix bank conflicts when the underlying memory layout causes all threads to map to the same bank regardless of load instruction style. Also avoid adding more num_stages when the kernel is already grid-size limited (waves_per_SM < 0.1), as additional pipeline stages increase smem usage and warp stall cycles without adding parallelism.

---

## Iteration 4

**Best so far**: 1.5157× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_n_to_double_grid_and_cut_smem_bank_conflicts | 1.490× | 1.761× | ✓ | **✓ best** |
| B | set_block_k16_even_k_true_to_eliminate_masked_loop_and_smem_bank_conflicts | 1.078× | 1.043× | ✓ |  |
| C | reduce_block_k_to_eliminate_masked_loop_and_smem_bank_conflicts | 1.065× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_n_to... | Δ | set_block_k16_eve... | Δ | reduce_block_k_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11550 ns | 6560 ns | -4990 ns (-43.2%) | 11070 ns | -480 ns (-4.2%) | 11680 ns | +130 ns (+1.1%) |
| SM Throughput (%) | 1.64 % | 1.31 % | -0.33 pp | 1.08 % | -0.56 pp | 1.09 % | -0.55 pp |
| Memory Throughput (%) | 13.34 % | 5.95 % | -7.39 pp | 5.31 % | -8.03 pp | 5.36 % | -7.98 pp |
| DRAM Throughput (%) | 0.93 % | 1.63 % | +0.70 pp | 0.96 % | +0.03 pp | 0.91 % | -0.02 pp |
| Achieved Occupancy (%) | 5.89 % | 6.23 % | +0.34 pp | 6.25 % | +0.36 pp | 6.25 % | +0.36 pp |
| Registers/Thread | 56.00 register/thread | 44.00 register/thread | -12 | 40.00 register/thread | -16 | 40.00 register/thread | -16 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 89.96 % | 91.64 % | +1.68 pp | 93.22 % | +3.26 pp | 93.22 % | +3.26 pp |
| IPC Active | 0.37 inst/cycle | 0.30 inst/cycle | -0.07 | 0.28 inst/cycle | -0.09 | 0.27 inst/cycle | -0.10 |


### Key Learnings

#### ✅ Plan C: `reduce_block_k_to_eliminate_masked_loop_and_smem_bank_conflicts` (1.516×)

When BLOCK_K does not evenly divide K, the final K-loop iteration processes mostly masked (invalid) elements, wasting compute and memory bandwidth. Choosing a BLOCK_K that exactly divides K (EVEN_K=True) eliminates mask computation and all partial-tile overhead. Simultaneously, reducing BLOCK_K shrinks the shared memory tile used by tl.dot, which lowers the shared memory footprint per block (33 KB → 17 KB), increases Block Limit Shared Mem (6 → 11), raises theoretical occupancy (37.5% → 62.5%), and reduces shared memory bank conflict severity (11.8-way → 6.6-way conflicts, MIO stall cycles 4.0 → 2.7). The combined effect is fewer executed instructions (143K → 93K), lower register pressure (56 → 41 regs/thread), and a 1.52x end-to-end speedup.

> **Applicable when**: SharedMemoryConflicts NCU rule fires with high bank-conflict-way (>8) and conflict wavefront % (>60%); BLOCK_K does not evenly divide K (EVEN_K=False); shared memory per block is large enough to limit Block Limit Shared Mem to a low value (<8); a smaller BLOCK_K value exists that exactly divides K; kernel is a GEMM or GEMM-like reduction loop in Triton.

> **Avoid when**: BLOCK_K already evenly divides K (EVEN_K already True) so no masking overhead exists; reducing BLOCK_K would increase the number of K-loop iterations to the point where loop overhead dominates; the kernel is already compute-bound on tensor cores and reducing tile size would hurt utilization; BLOCK_K is already at the minimum (e.g. 16) and further reduction is not possible.

---

## Iteration 5

**Best so far**: 1.7607× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_num_stages_to_hide_l1tex_latency | 1.524× | 1.037× | ✓ |  |
| C | increase_num_stages_to_hide_l1tex_latency_via_software_pipelining | 1.542× | 1.037× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11550 ns | 11140 ns | -410 ns (-3.5%) | 11140 ns | -410 ns (-3.5%) |
| SM Throughput (%) | 1.64 % | 1.06 % | -0.58 pp | 1.12 % | -0.52 pp |
| Memory Throughput (%) | 13.34 % | 5.18 % | -8.16 pp | 5.46 % | -7.88 pp |
| DRAM Throughput (%) | 0.93 % | 0.95 % | +0.02 pp | 0.96 % | +0.03 pp |
| Achieved Occupancy (%) | 5.89 % | 6.25 % | +0.36 pp | 6.25 % | +0.36 pp |
| Registers/Thread | 56.00 register/thread | 40.00 register/thread | -16 | 40.00 register/thread | -16 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 89.96 % | 93.02 % | +3.06 pp | 93.10 % | +3.14 pp |
| IPC Active | 0.37 inst/cycle | 0.27 inst/cycle | -0.10 | 0.26 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  11550 ns  (1.000×)
Iter 1     10370 ns  (1.114×)  — increase_tile_sizes_and_autotune_for_small_gemm_epilogue
Iter 3     7620 ns  (1.516×)  — reduce_block_k_to_eliminate_masked_loop_and_smem_bank_conflicts
Iter 4     6560 ns  (1.761×)  — reduce_block_n_to_double_grid_and_cut_smem_bank_conflicts

Final best: 11140 ns  (1.0368×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=4, num_stages=1),
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=4, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=4),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_16(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
-    EVEN_K : tl.constexpr = False
+    EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 2
+    BLOCK_N : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,26 +65,23 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 864*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # B tile: load as [BLOCK_N, BLOCK_K] with contiguous K (stride=1)
+        # B[k, n] = B + k*stride_bk + n*stride_bn = B + k*1 + n*144
+        b_k_vals = offs_k[None, :] + (k_idx * BLOCK_K)  # shape [1, BLOCK_K]
+        b_n_vals = offs_b_n[:, None]                      # shape [BLOCK_N, 1]
+        b = tl.load(
+            B + (b_k_vals + 144 * b_n_vals)  # shape [BLOCK_N=128, BLOCK_K=16], inner stride=1
+        )
+        # tl.trans(b): [BLOCK_N=128, BLOCK_K=16] -> [BLOCK_K=16, BLOCK_N=128] for tl.dot
+        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -106,8 +110,6 @@

     tmp15 = tl.full([1], 0, tl.int32)
     tmp16 = tl.maximum(tmp15, tmp14)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 864*idx_m, [BLOCK_M, BLOCK_N])), tmp16, mask)
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
