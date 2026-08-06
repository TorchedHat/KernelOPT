# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_clone_permute_t_view_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55330 ns — **Best latency**: 51420 ns (**1.0760×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_matrix_load_to_eliminate_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_clone_permute_t_view_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_8_triton_tem_fused_addmm_clone_p/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (112,1,1), Waves/SM: 0.42
- **Registers/thread**: 128
- **Duration**: 55330 ns
- **SM Throughput**: 10.54 % | **Memory**: 76.92 % | **DRAM**: 0.70 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.52 inst/cycle | **No Eligible**: 87.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 1.007× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_autotune_tile_sizes | 0.832× | 0.832× | ✓ |  |
| C | add_autotune_with_eviction_policies_and_wider_tiles | 0.830× | 0.831× | ✓ |  |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.831× | 0.833× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_shared_memory... | Δ | add_autotune_with... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 54940 ns | -390 ns (-0.7%) | 66530 ns | +11200 ns (+20.2%) | 66560 ns | +11230 ns (+20.3%) | 66400 ns | +11070 ns (+20.0%) |
| SM Throughput (%) | 10.54 % | 10.55 % | +0.01 pp | 16.30 % | +5.76 pp | 16.26 % | +5.72 pp | 16.25 % | +5.71 pp |
| Memory Throughput (%) | 76.92 % | 77.03 % | +0.11 pp | 64.03 % | -12.89 pp | 63.82 % | -13.10 pp | 63.81 % | -13.11 pp |
| DRAM Throughput (%) | 0.70 % | 0.71 % | +0.01 pp | 0.58 % | -0.12 pp | 0.58 % | -0.12 pp | 0.58 % | -0.12 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.42 | +0.00 | 0.28 | -0.14 | 0.28 | -0.14 | 0.28 | -0.14 |
| No Eligible (%) | 87.08 % | 87.05 % | -0.03 pp | 79.64 % | -7.44 pp | 79.69 % | -7.39 pp | 79.65 % | -7.43 pp |
| IPC Active | 0.52 inst/cycle | 0.52 inst/cycle | +0.00 | 0.81 inst/cycle | +0.29 | 0.81 inst/cycle | +0.29 | 0.81 inst/cycle | +0.29 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | widen_block_n_with_proper_autotune_and_grid_lambda | 0.853× | 0.812× | ✓ |  |
| B | autotune_tile_sizes_as_kernel_params_with_dynamic_grid | 0.521× | 0.112× | ✓ |  |
| C | autotune_tile_sizes_as_kernel_params_with_dynamic_grid | 1.058× | 1.079× | ✓ | **✓ best** |
| D | proper_autotune_tile_sizes_as_kernel_constexpr_params_with_lambda_grid | 0.620× | 0.553× | ✓ |  |


### Metric Comparison

| Metric | Baseline | widen_block_n_wit... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | proper_autotune_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 68100 ns | +12770 ns (+23.1%) | 495940 ns | +440610 ns (+796.3%) | 51300 ns | -4030 ns (-7.3%) | 100100 ns | +44770 ns (+80.9%) |
| SM Throughput (%) | 10.54 % | 15.91 % | +5.37 pp | 11.40 % | +0.86 pp | 11.15 % | +0.61 pp | 21.16 % | +10.62 pp |
| Memory Throughput (%) | 76.92 % | 62.43 % | -14.49 pp | 21.95 % | -54.97 pp | 77.74 % | +0.82 pp | 48.22 % | -28.70 pp |
| DRAM Throughput (%) | 0.70 % | 0.58 % | -0.12 pp | 11.60 % | +10.90 pp | 0.75 % | +0.05 pp | 0.39 % | -0.31 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 208.00 register/thread | +80 | 255.00 register/thread | +127 | 96.00 register/thread | -32 | 255.00 register/thread | +127 |
| Waves/SM | 0.42 | 0.85 | +0.43 | 0.85 | +0.43 | 0.21 | -0.21 | 0.42 | +0.00 |
| No Eligible (%) | 87.08 % | 79.83 % | -7.25 pp | 91.76 % | +4.68 pp | 86.30 % | -0.78 pp | 72.59 % | -14.49 pp |
| IPC Active | 0.52 inst/cycle | 0.81 inst/cycle | +0.29 | 0.33 inst/cycle | -0.19 | 0.55 inst/cycle | +0.03 | 1.10 inst/cycle | +0.58 |

---

## Iteration 3

**Best so far**: 1.0786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expose_tile_sizes_as_kernel_params_with_proper_autotune_and_lambda_grid | 1.048× | 1.061× | ✓ |  |
| C | reduce_register_spill_via_smaller_tiles_and_proper_autotune | 1.049× | 1.063× | ✓ |  |
| D | reduce_register_pressure_by_shrinking_tile_sizes_to_eliminate_local_memory_spill | 1.045× | 1.065× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expose_tile_sizes... | Δ | reduce_register_s... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 52130 ns | -3200 ns (-5.8%) | 52030 ns | -3300 ns (-6.0%) | 51970 ns | -3360 ns (-6.1%) |
| SM Throughput (%) | 10.54 % | 11.13 % | +0.59 pp | 11.07 % | +0.53 pp | 11.08 % | +0.54 pp |
| Memory Throughput (%) | 76.92 % | 76.88 % | -0.04 pp | 76.41 % | -0.51 pp | 76.52 % | -0.40 pp |
| DRAM Throughput (%) | 0.70 % | 0.73 % | +0.03 pp | 0.73 % | +0.03 pp | 0.73 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 80.00 register/thread | -48 | 80.00 register/thread | -48 | 80.00 register/thread | -48 |
| Waves/SM | 0.42 | 0.14 | -0.28 | 0.14 | -0.28 | 0.14 | -0.28 |
| No Eligible (%) | 87.08 % | 86.36 % | -0.72 pp | 86.37 % | -0.71 pp | 86.38 % | -0.70 pp |
| IPC Active | 0.52 inst/cycle | 0.55 inst/cycle | +0.03 | 0.55 inst/cycle | +0.03 | 0.54 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.0786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_to_eliminate_shared_memory_bank_conflicts | 1.062× | 1.076× | ✓ | **✓ best** |
| B | fix_bank_conflicts_via_block_ptr_with_correct_order_for_b_matrix | 1.051× | 1.067× | ✓ |  |
| C | fix_b_matrix_bank_conflicts_via_transposed_smem_load | 1.054× | 1.060× | ✓ |  |
| D | add_triton_autotune | 1.046× | 1.061× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_bank_conflict... | Δ | fix_b_matrix_bank... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 51420 ns | -3910 ns (-7.1%) | 51870 ns | -3460 ns (-6.3%) | 52190 ns | -3140 ns (-5.7%) | 52160 ns | -3170 ns (-5.7%) |
| SM Throughput (%) | 10.54 % | 11.23 % | +0.69 pp | 11.32 % | +0.78 pp | 11.08 % | +0.54 pp | 11.12 % | +0.58 pp |
| Memory Throughput (%) | 76.92 % | 77.53 % | +0.61 pp | 76.96 % | +0.04 pp | 76.54 % | -0.38 pp | 76.80 % | -0.12 pp |
| DRAM Throughput (%) | 0.70 % | 0.74 % | +0.04 pp | 0.73 % | +0.03 pp | 0.73 % | +0.03 pp | 0.73 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 94.00 register/thread | -34 | 80.00 register/thread | -48 | 80.00 register/thread | -48 | 80.00 register/thread | -48 |
| Waves/SM | 0.42 | 0.17 | -0.25 | 0.14 | -0.28 | 0.14 | -0.28 | 0.14 | -0.28 |
| No Eligible (%) | 87.08 % | 86.18 % | -0.90 pp | 86.11 % | -0.97 pp | 86.32 % | -0.76 pp | 86.35 % | -0.73 pp |
| IPC Active | 0.52 inst/cycle | 0.55 inst/cycle | +0.03 | 0.56 inst/cycle | +0.04 | 0.55 inst/cycle | +0.03 | 0.54 inst/cycle | +0.02 |


### Key Learnings

#### ✅ Plan C: `fix_b_matrix_bank_conflicts_via_transposed_smem_load` (1.060×)

Restructuring the B matrix load from shape [BLOCK_K, BLOCK_N] with stride-512 in K to shape [BLOCK_N, BLOCK_K] with stride-1 in K followed by tl.trans() was predicted to eliminate 17-way shared memory bank conflicts (NCU SharedMemoryConflicts rule: 77.42% speedup potential). However, the bank conflict metrics were completely unchanged after the rewrite: SharedMemoryConflicts still reported 6,422,528 bank conflicts at 82.28% wavefront waste, L1/TEX throughput remained at ~94%, and elapsed cycles were nearly identical (77,905 vs 78,219). The 1.06x speedup is within measurement noise. The likely explanation is that Triton's internal shared memory staging for tl.dot re-introduces the conflicting access pattern regardless of how the explicit load is structured — tl.trans() performs a register-level permutation but Triton's software pipeline may still stage the B tile through shared memory with the same conflicting layout when feeding tl.dot.

> **Applicable when**: Never — this is a negative pattern. The transposed B load + tl.trans() approach does not resolve shared memory bank conflicts when the conflicts originate from Triton's internal shared memory staging for tl.dot rather than from the explicit user-level load pattern.

> **Avoid when**: When SharedMemoryConflicts NCU rule fires on a Triton GEMM kernel and the conflicts are attributed to the B matrix load: restructuring the explicit load shape and using tl.trans() does not help if Triton's tl.dot software pipeline re-stages the tile through shared memory with the same conflicting layout. The bank conflict pattern is determined by how Triton internally lays out the B tile in shared memory for tl.dot, not by the shape of the user-level tl.load call. Avoid spending optimization budget on this approach when the NCU rule points to shared load wavefronts associated with tl.dot internals.

---

## Iteration 5

**Best so far**: 1.0786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_kernel_scheduling_to_eliminate_sm_underutilization | 1.026× | 1.040× | ✓ |  |
| B | fix_b_matrix_bank_conflicts_via_explicit_smem_padding | 1.040× | 1.054× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | fix_b_matrix_bank... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 53180 ns | -2150 ns (-3.9%) | 52480 ns | -2850 ns (-5.2%) |
| SM Throughput (%) | 10.54 % | 10.10 % | -0.44 pp | 11.14 % | +0.60 pp |
| Memory Throughput (%) | 76.92 % | 69.47 % | -7.45 pp | 76.16 % | -0.76 pp |
| DRAM Throughput (%) | 0.70 % | 0.72 % | +0.02 pp | 0.73 % | +0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 96.00 register/thread | -32 | 96.00 register/thread | -32 |
| Waves/SM | 0.42 | 0.17 | -0.25 | 0.17 | -0.25 |
| No Eligible (%) | 87.08 % | 86.65 % | -0.43 pp | 86.37 % | -0.71 pp |
| IPC Active | 0.52 inst/cycle | 0.53 inst/cycle | +0.01 | 0.55 inst/cycle | +0.03 |


### Key Learnings

#### ✅ Plan A: `persistent_kernel_scheduling_to_eliminate_sm_underutilization` (1.040×)

Converting a one-block-per-tile launch to a persistent kernel (each SM loops over multiple tiles via `while tile_id < num_tiles: tile_id += num_programs`) was predicted to eliminate idle SMs and raise achieved occupancy from 6.25% toward theoretical 31.25%. However, all occupancy metrics remained completely unchanged after the rewrite: achieved occupancy stayed at 6.25%, active warps per scheduler stayed at 1.0, waves_per_sm stayed at 0.17, and the grid size remained 112. The persistent loop added control-flow overhead (branch instructions rose from 8,000 to 8,768) and warp cycles per issued instruction worsened from 7.25 to 7.49, causing elapsed cycles to increase from 77,067 to 79,731. The 1.04x reported speedup is likely measurement noise masking a slight regression in raw kernel time. The root cause is that with only 104 output tiles and 132 SMs, the workload is fundamentally too small to fill the GPU regardless of launch strategy — the persistent loop does not create more parallelism, it only serializes tiles within each SM.

> **Applicable when**: Never for this workload pattern. Persistent kernels are beneficial when num_tiles >> num_sm (many waves) and the bottleneck is kernel launch overhead or tail-effect imbalance across waves. They do NOT help when num_tiles < num_sm (as here: 104 tiles < 132 SMs) because the workload is structurally too small to fill the GPU regardless of scheduling strategy.

> **Avoid when**: When waves_per_sm < 1.0 and num_tiles < num_sm: the persistent loop serializes tiles within each SM without creating additional parallelism, adding branch overhead while leaving achieved occupancy, active warps per scheduler, and No Eligible percentage completely unchanged. Also avoid when the dominant bottleneck is SharedMemoryConflicts (17-way bank conflicts, 82%+ wavefront waste) — persistent scheduling does not address memory access pattern issues. Do not apply when the kernel already has high register pressure (94+ registers/thread) as the while-loop adds more branch instructions and may worsen warp stall cycles.

---

## Cumulative Progress

```
Baseline:  55330 ns  (1.000×)
Iter 2     51300 ns  (1.079×)  — autotune_tile_sizes_as_kernel_params_with_dynamic_grid

Final best: 51420 ns  (1.0760×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,28 @@

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
 def triton_tem_fused_addmm_clone_permute_t_view_8(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_M : tl.constexpr = 32
+    BLOCK_N : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,23 +72,20 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 512*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 512*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # B has stride_bk=1 (K contiguous), stride_bn=512 (N stride)
+        # B[k, n] = B_ptr + k*1 + n*512 -- load as natural [BLOCK_K, BLOCK_N] row-major tile
+        b_k_offs = offs_k[:, None] + (k_idx * BLOCK_K)  # shape [BLOCK_K, 1]
+        b = tl.load(B + b_k_offs + 512 * offs_b_n[None, :])  # shape [BLOCK_K, BLOCK_N]
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +97,6 @@

     # inductor generates a suffix
     xindex = idx_n + 512*idx_m
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
