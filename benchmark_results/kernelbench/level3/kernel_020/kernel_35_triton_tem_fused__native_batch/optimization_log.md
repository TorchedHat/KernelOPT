# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55870 ns — **Best latency**: 54370 ns (**1.0276×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`add_autotune_with_eviction_policies`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_35_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 94
- **Duration**: 55870 ns
- **SM Throughput**: 7.96 % | **Memory**: 53.86 % | **DRAM**: 0.92 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 1.026× | 0.902× | ✓ |  |
| C | add_autotune_with_eviction_policies | 1.031× | 1.028× | ✓ | **✓ best** |
| D | autotune_tile_sizes_and_warps_for_gemm | 0.551× | 0.537× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_autotune_with... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55870 ns | 61920 ns | +6050 ns (+10.8%) | 54370 ns | -1500 ns (-2.7%) | 104100 ns | +48230 ns (+86.3%) |
| SM Throughput (%) | 7.96 % | 6.30 % | -1.66 pp | 7.98 % | +0.02 pp | 13.95 % | +5.99 pp |
| Memory Throughput (%) | 53.86 % | 45.32 % | -8.54 pp | 51.85 % | -2.01 pp | 54.22 % | +0.36 pp |
| DRAM Throughput (%) | 0.92 % | 0.83 % | -0.09 pp | 0.94 % | +0.02 pp | 0.49 % | -0.43 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 96.00 register/thread | +2 | 64.00 register/thread | -30 | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.06 | -0.14 | 0.08 | -0.12 | 0.15 | -0.05 |
| No Eligible (%) | 86.35 % | 78.49 % | -7.86 pp | 86.32 % | -0.03 pp | 76.42 % | -9.93 pp |
| IPC Active | 0.55 inst/cycle | 0.43 inst/cycle | -0.12 | 0.55 inst/cycle | +0.00 | 0.94 inst/cycle | +0.39 |

---

## Iteration 2

**Best so far**: 1.0276× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_tile_size_change | 0.475× | 0.461× | ✓ |  |
| B | promote_tile_sizes_to_autotune_params_fix_bank_conflicts | 0.551× | 0.421× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_via_b_tile_transpose | 0.549× | 0.533× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | promote_tile_size... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55870 ns | 121250 ns | +65380 ns (+117.0%) | 132770 ns | +76900 ns (+137.6%) | 104770 ns | +48900 ns (+87.5%) |
| SM Throughput (%) | 7.96 % | 11.86 % | +3.90 pp | 10.64 % | +2.68 pp | 13.91 % | +5.95 pp |
| Memory Throughput (%) | 53.86 % | 46.47 % | -7.39 pp | 23.76 % | -30.10 pp | 54.05 % | +0.19 pp |
| DRAM Throughput (%) | 0.92 % | 0.43 % | -0.49 pp | 0.39 % | -0.53 pp | 0.49 % | -0.43 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 168.00 register/thread | +74 | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.20 | +0.00 | 0.10 | -0.10 | 0.15 | -0.05 |
| No Eligible (%) | 86.35 % | 79.70 % | -6.65 pp | 61.34 % | -25.01 pp | 76.49 % | -9.86 pp |
| IPC Active | 0.55 inst/cycle | 0.81 inst/cycle | +0.26 | 0.77 inst/cycle | +0.22 | 0.94 inst/cycle | +0.39 |


### Key Learnings

#### ❌ Plan C: `fix_shared_memory_bank_conflicts_via_b_tile_transpose` (0.533×)

Applying tl.trans() on the B tile to fix shared memory bank conflicts was bundled with doubling BLOCK_M and BLOCK_N (32→64) and adding num_warps=16 configs. This caused registers_per_thread to double from 64 to 128 and dynamic shared memory per block to double from 16 to 33 KB, collapsing theoretical occupancy from 50% to 25% (Block Limit Registers dropped from 8 to 4 blocks/SM). Although per-instruction efficiency improved (IPC active 0.55→0.94, warp cycles/instruction 7.3→4.3), the 3.3x increase in total executed instructions (3.45M→11.5M) caused duration to nearly double (54→105 µs). The bank conflict rate remained unchanged (~12-way, ~77% conflicted wavefronts), indicating the tl.trans() approach did not actually eliminate the shared memory bank conflicts. When a kernel is already register-limited (Block Limit Registers is the binding occupancy constraint), increasing tile sizes or adding large num_warps configs will further reduce occupancy and cause regression even if per-warp efficiency improves.

> **Applicable when**: Never — this specific combination caused regression. The tl.trans() approach alone (without tile size increase) might be worth isolating, but as applied here it is harmful.

> **Avoid when**: When the kernel is already register-limited (Block Limit Registers is the binding occupancy constraint, e.g. 8 blocks/SM at 64 registers/thread). Doubling tile sizes (BLOCK_M, BLOCK_N 32→64) in this regime doubles register pressure (64→128 regs/thread), halves theoretical occupancy (50%→25%), and increases total instruction count by 3x, causing ~2x duration regression even though per-warp efficiency improves. Additionally, tl.trans() did not eliminate the 12-way shared memory bank conflicts in this case — the bank conflict rate remained ~77% of wavefronts after the change, suggesting the root cause was not purely the B load access pattern or that tl.trans() does not bypass shared memory for this tile size.

---

## Iteration 3

**Best so far**: 1.0276× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_halve_smem_and_fix_bank_conflicts | 0.508× | 0.494× | ✓ |  |
| B | promote_tile_sizes_to_true_autotune_kernel_params | 1.026× | 1.026× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tiles_as_autotune_params | 1.026× | 0.898× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | promote_tile_size... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55870 ns | 113150 ns | +57280 ns (+102.5%) | 54460 ns | -1410 ns (-2.5%) | 62180 ns | +6310 ns (+11.3%) |
| SM Throughput (%) | 7.96 % | 13.09 % | +5.13 pp | 8.00 % | +0.04 pp | 6.30 % | -1.66 pp |
| Memory Throughput (%) | 53.86 % | 49.98 % | -3.88 pp | 51.96 % | -1.90 pp | 45.36 % | -8.50 pp |
| DRAM Throughput (%) | 0.92 % | 0.45 % | -0.47 pp | 0.94 % | +0.02 pp | 0.82 % | -0.10 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 64.00 register/thread | -30 | 96.00 register/thread | +2 |
| Waves/SM | 0.20 | 0.15 | -0.05 | 0.08 | -0.12 | 0.06 | -0.14 |
| No Eligible (%) | 86.35 % | 77.93 % | -8.42 pp | 86.35 % | +0.00 pp | 78.53 % | -7.82 pp |
| IPC Active | 0.55 inst/cycle | 0.88 inst/cycle | +0.33 | 0.55 inst/cycle | +0.00 | 0.43 inst/cycle | -0.12 |


### Key Learnings

#### ❌ Plan A: `reduce_block_k_to_halve_smem_and_fix_bank_conflicts` (0.494×)

Reducing BLOCK_K from 32 to 16 halved dynamic shared memory per block (33KB→16KB), successfully raising Block Limit Shared Mem from 4 to 7 blocks/SM. However, theoretical and achieved occupancy remained unchanged at 25% and 6.25% respectively, because registers per thread (128) remained the binding constraint (Block Limit Registers=4). With occupancy unchanged, the doubled K-loop iteration count (30→60) added ~2% more instructions and worsened warp cycles per instruction (4.26→4.53), causing elapsed cycles to increase from 157k to 170k. The SharedMemoryConflicts rule remained triggered with the same 12-way average conflict degree and ~77% conflicted wavefronts, indicating that halving BLOCK_K did not materially change the bank conflict pattern for this access layout.

> **Applicable when**: Never — this is a negative result. Reducing BLOCK_K to shrink shared memory footprint does not help when registers per thread are the binding occupancy constraint (Block Limit Registers < Block Limit Shared Mem).

> **Avoid when**: When registers per thread are the binding occupancy limiter (Block Limit Registers ≤ Block Limit Shared Mem): halving BLOCK_K reduces shared memory pressure but cannot increase occupancy beyond the register ceiling. The doubled loop iteration count then adds overhead (more instructions, worse warp cycles/instruction) without any compensating benefit. In this kernel, 128 registers/thread with BLOCK_M=64, BLOCK_N=64 fixes Block Limit Registers=4 regardless of BLOCK_K. The bank conflict degree also did not improve (still 12-way, ~77% conflicted wavefronts), suggesting the conflict pattern is driven by the BLOCK_M/BLOCK_N tile dimensions and the transposed access stride, not by BLOCK_K.

---

## Cumulative Progress

```
Baseline:  55870 ns  (1.000×)
Iter 1     54370 ns  (1.028×)  — add_autotune_with_eviction_policies

Final best: 54370 ns  (1.0276×)
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
+        triton.Config({}, num_warps=16, num_stages=3),
+        triton.Config({}, num_warps=16, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
+    EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_M : tl.constexpr = 64
+    BLOCK_N : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -60,24 +74,24 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 960*idx_m
         a = tl.load(A + (xindex), mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 160*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 960*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
+        # Coalesced transposed B load: load [BLOCK_N, BLOCK_K] then tl.trans to [BLOCK_K, BLOCK_N]
+        offs_k_cur = offs_k + k_idx * BLOCK_K   # absolute K offsets, shape [BLOCK_K]
+        b_n_idx = offs_b_n[:, None]              # shape [BLOCK_N, 1]
+        b_k_idx = offs_k_cur[None, :]            # shape [1, BLOCK_K]
+        b_mask_t = b_k_idx < K                   # shape broadcastable to [BLOCK_N, BLOCK_K]
+        # Load [BLOCK_N, BLOCK_K]: stride_bn=960 (outer), stride_bk=1 (inner, contiguous)
+        b_tile = tl.load(B + (b_n_idx * 960 + b_k_idx), mask=b_mask_t, other=0.0)
+        b = tl.trans(b_tile)                     # shape [BLOCK_K, BLOCK_N] for tl.dot
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +103,6 @@

     # inductor generates a suffix
     xindex = idx_n + 160*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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
