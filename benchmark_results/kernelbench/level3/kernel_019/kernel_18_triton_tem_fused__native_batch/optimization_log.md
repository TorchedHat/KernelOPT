# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_18` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 58400 ns — **Best latency**: 58140 ns (**1.0045×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_matrix_broadcast_to_with_explicit_2d_indexing_only`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_18`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_18_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.24
- **Registers/thread**: 128
- **Duration**: 58400 ns
- **SM Throughput**: 21.26 % | **Memory**: 83.11 % | **DRAM**: 1.09 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 77.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 0.999× | ✓ |  |
| B | tma_loads_with_autotune_tile_sizes | 0.941× | 0.590× | ✓ |  |
| C | add_eviction_policies_to_streaming_loads | 1.001× | 1.004× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 1.000× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | add_eviction_poli... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58400 ns | 58460 ns | +60 ns (+0.1%) | 99040 ns | +40640 ns (+69.6%) | 58180 ns | -220 ns (-0.4%) | 58140 ns | -260 ns (-0.4%) |
| SM Throughput (%) | 21.26 % | 21.31 % | +0.05 pp | 13.38 % | -7.88 pp | 21.28 % | +0.02 pp | 21.24 % | -0.02 pp |
| Memory Throughput (%) | 83.11 % | 83.30 % | +0.19 pp | 92.22 % | +9.11 pp | 83.16 % | +0.05 pp | 83.17 % | +0.06 pp |
| DRAM Throughput (%) | 1.09 % | 1.09 % | +0.00 pp | 0.64 % | -0.45 pp | 1.09 % | +0.00 pp | 1.10 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.49 % | +6.24 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 80.00 register/thread | -48 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.32 | +0.08 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.04 % | 77.05 % | +0.01 pp | 85.88 % | +8.84 pp | 77.01 % | -0.03 pp | 77.02 % | -0.02 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.56 inst/cycle | -0.36 | 0.92 inst/cycle | +0.00 | 0.91 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0038× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_transposed_load_to_eliminate_shared_memory_bank_conflicts | 1.000× | 1.001× | ✓ |  |
| C | autotune_num_stages_pipelining_to_hide_mio_stalls_and_fix_grid_utilization | 0.996× | 1.006× | ✓ |  |
| D | fix_b_matrix_block_ptr_order_to_eliminate_bank_conflicts | 0.943× | 0.592× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_tran... | Δ | autotune_num_stag... | Δ | fix_b_matrix_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58400 ns | 58340 ns | -60 ns (-0.1%) | 58020 ns | -380 ns (-0.7%) | 98720 ns | +40320 ns (+69.0%) |
| SM Throughput (%) | 21.26 % | 21.19 % | -0.07 pp | 21.31 % | +0.05 pp | 13.38 % | -7.88 pp |
| Memory Throughput (%) | 83.11 % | 82.82 % | -0.29 pp | 83.30 % | +0.19 pp | 92.21 % | +9.10 pp |
| DRAM Throughput (%) | 1.09 % | 1.09 % | +0.00 pp | 1.10 % | +0.01 pp | 0.64 % | -0.45 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 12.49 % | +6.24 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 80.00 register/thread | -48 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.32 | +0.08 |
| No Eligible (%) | 77.04 % | 77.00 % | -0.04 pp | 77.11 % | +0.07 pp | 85.89 % | +8.85 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.92 inst/cycle | +0.00 | 0.56 inst/cycle | -0.36 |

---

## Iteration 3

**Best so far**: 1.0038× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_broadcast_to_with_explicit_2d_indexing_only | 1.002× | 1.004× | ✓ | **✓ best** |
| B | eliminate_tl_trans_by_loading_b_as_k_by_n_directly | 1.000× | 1.007× | ✓ |  |
| C | pretranspose_b_matrix_to_eliminate_shared_memory_bank_conflicts | 0.940× | 0.590× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_broa... | Δ | eliminate_tl_tran... | Δ | pretranspose_b_ma... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58400 ns | 58140 ns | -260 ns (-0.4%) | 57980 ns | -420 ns (-0.7%) | 98910 ns | +40510 ns (+69.4%) |
| SM Throughput (%) | 21.26 % | 21.27 % | +0.01 pp | 21.29 % | +0.03 pp | 13.36 % | -7.90 pp |
| Memory Throughput (%) | 83.11 % | 83.18 % | +0.07 pp | 83.20 % | +0.09 pp | 92.14 % | +9.03 pp |
| DRAM Throughput (%) | 1.09 % | 1.09 % | +0.00 pp | 1.10 % | +0.01 pp | 0.64 % | -0.45 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 12.49 % | +6.24 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 80.00 register/thread | -48 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.32 | +0.08 |
| No Eligible (%) | 77.04 % | 77.02 % | -0.02 pp | 77.01 % | -0.03 pp | 85.90 % | +8.86 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.92 inst/cycle | +0.00 | 0.56 inst/cycle | -0.36 |


### Key Learnings

#### ❌ Plan C: `pretranspose_b_matrix_to_eliminate_shared_memory_bank_conflicts` (0.590×)

Physically pre-transposing the B matrix in the host wrapper (reshape + .t().contiguous()) and switching to tl.make_block_ptr with order=(1,0) for the transposed layout produced the same regression as prior block_ptr attempts: bank conflict wavefronts doubled (7.3M → 14.7M), average conflict way increased (12-way → 17-way), warp cycles per issued instruction tripled (4.35 → 14.18), and kernel duration nearly doubled (58µs → 99µs). Despite the physical memory layout being correct (stride_bk=1 contiguous after transpose), the tl.make_block_ptr staging into shared memory still generates severe conflicts, indicating the conflict source is in the shared memory access pattern during tl.dot, not the global load pattern. The scalar pointer arithmetic baseline (58µs) remains superior to any block_ptr variant tried across three iterations.

> **Applicable when**: Never — this direction has now failed three times with statistically identical profiling outcomes regardless of block_ptr order parameter or physical memory layout of B.

> **Avoid when**: Do not apply tl.make_block_ptr (in any order configuration) or host-side B pre-transposition as a fix for shared memory bank conflicts in this kernel. Three iterations (order=(1,0), order=(0,1), and pre-transposed B with order=(1,0)) all produced identical regressions: bank conflict wavefronts doubled to ~14.7M, warp stall cycles tripled, and duration nearly doubled to ~99µs. The scalar pointer arithmetic baseline at 58µs is consistently superior. The bank conflicts appear to originate from the shared memory access pattern during tl.dot accumulation, not from the global load pattern, and cannot be fixed by changing block_ptr order or physical B layout alone.

---

## Iteration 4

**Best so far**: 1.0045× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | pad_shared_memory_via_nonpower2_block_k_to_eliminate_bank_conflicts | 0.457× | 0.431× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pad_shared_memory... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 58400 ns | 135360 ns | +76960 ns (+131.8%) |
| SM Throughput (%) | 21.26 % | 19.19 % | -2.07 pp |
| Memory Throughput (%) | 83.11 % | 77.94 % | -5.17 pp |
| DRAM Throughput (%) | 1.09 % | 0.47 % | -0.62 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 255.00 register/thread | +127 |
| Waves/SM | 0.24 | 0.48 | +0.24 |
| No Eligible (%) | 77.04 % | 79.25 % | +2.21 pp |
| IPC Active | 0.92 inst/cycle | 0.83 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan B: `pad_shared_memory_via_nonpower2_block_k_to_eliminate_bank_conflicts` (0.431×)

Setting BLOCK_K=33 (a non-power-of-2 value) in Triton requires padding tl.arange to the next power-of-2 (64) and adding per-element masks on every load. This caused registers per thread to explode from 128 to 255 (hitting the hardware maximum), theoretical occupancy to drop from 25% to 12.5%, dynamic shared memory to double (32KB → 65KB), and executed instructions to more than double (9.8M → 20.5M). Critically, bank conflict wavefronts also doubled (7.3M → 14.7M) rather than decreasing, because the padded arange with masking does not actually change the shared memory layout seen by tl.dot — the conflict pattern is unchanged. The masking overhead and register spill completely dominate, producing a 2.3× slowdown. Do not use non-power-of-2 BLOCK_K values in Triton matmul kernels; the tl.arange power-of-2 constraint forces masking overhead that causes severe register pressure and instruction count inflation without fixing the underlying bank conflict pattern.

> **Applicable when**: Never applicable in Triton — non-power-of-2 BLOCK_K values always require padding tl.arange to the next power-of-2, which causes register explosion and instruction count doubling without fixing bank conflicts.

> **Avoid when**: Always avoid non-power-of-2 BLOCK_K (e.g., 33, 48) in Triton matmul kernels. The tl.arange power-of-2 constraint forces masking overhead that: (1) doubles instruction count, (2) pushes registers/thread to the hardware maximum (255), (3) halves theoretical occupancy, (4) doubles dynamic shared memory, and (5) does NOT reduce bank conflicts because tl.dot's shared memory staging layout is determined by the padded power-of-2 tile shape, not the masked BLOCK_K value. The classic '+1 padding' trick for shared memory bank conflicts cannot be applied via non-power-of-2 BLOCK_K in Triton.

---

## Iteration 5

**Best so far**: 1.0045× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | revert_block_k_to_power_of_2_to_eliminate_register_spill | 0.997× | 1.002× | ✓ |  |
| C | revert_block_k_to_power_of_2_and_add_streaming_eviction_policies | 0.996× | 1.003× | ✓ |  |
| D | revert_to_power_of_2_block_k_and_cap_registers_with_maxnreg | 0.901× | 0.908× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_block_k_to... | Δ | revert_block_k_to... | Δ | revert_to_power_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58400 ns | 58300 ns | -100 ns (-0.2%) | 58210 ns | -190 ns (-0.3%) | 64290 ns | +5890 ns (+10.1%) |
| SM Throughput (%) | 21.26 % | 21.28 % | +0.02 pp | 21.18 % | -0.08 pp | 19.25 % | -2.01 pp |
| Memory Throughput (%) | 83.11 % | 83.20 % | +0.09 pp | 82.81 % | -0.30 pp | 75.27 % | -7.84 pp |
| DRAM Throughput (%) | 1.09 % | 1.09 % | +0.00 pp | 1.09 % | +0.00 pp | 0.99 % | -0.10 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.04 % | 77.01 % | -0.03 pp | 77.00 % | -0.04 pp | 79.43 % | +2.39 pp |
| IPC Active | 0.92 inst/cycle | 0.91 inst/cycle | -0.01 | 0.92 inst/cycle | +0.00 | 0.82 inst/cycle | -0.10 |

---

## Cumulative Progress

```
Baseline:  58400 ns  (1.000×)
Iter 1     58180 ns  (1.004×)  — add_eviction_policies_to_streaming_loads
Iter 3     58140 ns  (1.004×)  — fix_b_matrix_broadcast_to_with_explicit_2d_indexing_only

Final best: 58140 ns  (1.0045×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_18(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    # BLOCK_K=33: non-power-of-32 breaks 12-way bank conflict stride in tl.dot shared memory staging.
+    # Classic padding trick: each row is 33 FP32 = 132 bytes, shifting bank alignment by 1 per row.
+    BLOCK_K : tl.constexpr = 33
+    # tl.arange() requires power-of-2 upper bound; 33 -> 64
+    BLOCK_K_POW2 : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -54,27 +72,28 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+    # Use BLOCK_K_POW2=64 for tl.arange (must be power of 2), mask to BLOCK_K=33
+    offs_k = tl.arange(0, BLOCK_K_POW2)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
+        # Mask: only use the first BLOCK_K=33 elements of the padded range
+        k_mask = offs_k < BLOCK_K
+
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
-        xindex = idx_n + 512*idx_m
-        a = tl.load(A + (xindex))
+        xindex = idx_n + 512 * idx_m
+        a_mask = (idx_m < M) & (a_k_idx_vals < K) & (k_mask[None, :])
+        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1024*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b_mask = (b_k_idx_vals < K) & (k_mask[:, None]) & (offs_b_n[None, :] < N)
+        b = tl.load(B + (b_k_idx_vals + 512 * offs_b_n[None, :]), mask=b_mask, other=0.0)
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,10 +103,8 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 1024*idx_m
+    xindex = idx_n + 1024 * idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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
