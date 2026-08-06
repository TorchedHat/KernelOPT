# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_gelu_t_view_56` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 329790 ns — **Best latency**: 351780 ns (**0.9375×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_gelu_t_view_56`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_56_triton_tem_fused_addmm_gelu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.18
- **Registers/thread**: 128
- **Duration**: 329790 ns
- **SM Throughput**: 16.64 % | **Memory**: 65.51 % | **DRAM**: 0.96 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.93 inst/cycle | **No Eligible**: 76.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.868× | 0.867× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.000× | 0.567× | ✓ | **✓ best** |
| C | autotune_tile_sizes_and_num_stages_to_reduce_bank_conflicts_and_register_pressure | 0.867× | 0.866× | ✓ |  |
| D | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 0.867× | 0.867× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 329790 ns | 380420 ns | +50630 ns (+15.4%) | 581250 ns | +251460 ns (+76.2%) | 381020 ns | +51230 ns (+15.5%) | 380610 ns | +50820 ns (+15.4%) |
| SM Throughput (%) | 16.64 % | 14.31 % | -2.33 pp | 9.92 % | -6.72 pp | 14.31 % | -2.33 pp | 14.31 % | -2.33 pp |
| Memory Throughput (%) | 65.51 % | 56.69 % | -8.82 pp | 70.19 % | +4.68 pp | 56.72 % | -8.79 pp | 56.68 % | -8.83 pp |
| DRAM Throughput (%) | 0.96 % | 0.83 % | -0.13 pp | 0.54 % | -0.42 pp | 0.83 % | -0.13 pp | 0.83 % | -0.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.50 % | +6.25 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 80.00 register/thread | -48 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.24 | +0.06 | 0.24 | +0.06 | 0.24 | +0.06 | 0.24 | +0.06 |
| No Eligible (%) | 76.85 % | 79.64 % | +2.79 pp | 86.30 % | +9.45 pp | 79.64 % | +2.79 pp | 79.63 % | +2.78 pp |
| IPC Active | 0.93 inst/cycle | 0.81 inst/cycle | -0.12 | 0.55 inst/cycle | -0.38 | 0.81 inst/cycle | -0.12 | 0.81 inst/cycle | -0.12 |


### Key Learnings

#### ❌ Plan A: `tf32_tensor_cores` (0.867×)

The plan intended to enable TF32 Tensor Cores to eliminate 12-way shared memory bank conflicts, but the actual diff only doubled BLOCK_K from 32 to 64 while leaving ALLOW_TF32=False unchanged. This doubled the shared memory allocation per block (32.77 KB → 65.54 KB), reducing Block Limit Shared Mem from 4 to 3 and lowering theoretical occupancy from 25% to 18.75%, while the bank-conflict count remained identical (~33M conflicts, 77.76% of wavefronts). The net effect was more stall cycles per instruction (4.32 → 4.91 cycles/instruction) and 15% longer duration, with no compute throughput improvement (16.64% → 14.31%).

> **Applicable when**: ALLOW_TF32 is actually flipped to True AND the resulting shared memory increase does not reduce Block Limit Shared Mem below the previous limit; bank conflicts are the dominant bottleneck (SharedMemoryConflicts rule, >50% wavefront overhead).

> **Avoid when**: ALLOW_TF32 remains False — in that case, increasing BLOCK_K only increases shared memory usage, reduces theoretical occupancy, and worsens stall cycles without eliminating bank conflicts. Also avoid when shared memory is already near the per-SM limit and increasing tile size would drop Block Limit Shared Mem.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_bank_conflicts_via_tl_make_block_ptr_tma_loads | 0.941× | 0.938× | ✓ | **✓ best** |
| B | fix_b_matrix_strided_access_to_eliminate_shared_memory_bank_conflicts | 0.868× | 0.866× | ✓ |  |
| D | autotune_tile_sizes_to_resolve_transposed_b_bank_conflicts | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_bank_co... | Δ | fix_b_matrix_stri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 329790 ns | 351780 ns | +21990 ns (+6.7%) | 380800 ns | +51010 ns (+15.5%) |
| SM Throughput (%) | 16.64 % | 15.88 % | -0.76 pp | 14.30 % | -2.34 pp |
| Memory Throughput (%) | 65.51 % | 61.45 % | -4.06 pp | 56.65 % | -8.86 pp |
| DRAM Throughput (%) | 0.96 % | 0.90 % | -0.06 pp | 0.83 % | -0.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.24 | +0.06 |
| No Eligible (%) | 76.85 % | 78.00 % | +1.15 pp | 79.65 % | +2.80 pp |
| IPC Active | 0.93 inst/cycle | 0.88 inst/cycle | -0.05 | 0.81 inst/cycle | -0.12 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  329790 ns  (1.000×)

Final best: 351780 ns  (0.9375×)
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
 def triton_tem_fused_addmm_gelu_t_view_56(arg_A, arg_B, out_ptr0):
@@ -44,37 +45,34 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointers for TMA loads — eliminates shared memory bank conflicts
+    # A: shape (M, K), strides (3072, 1), row-major
+    a_block_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(3072, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B: shape (K, N), strides (1, 3072), column-major — TMA handles this natively
+    b_block_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, 3072),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 3072*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 768*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 3072*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +84,6 @@

     # inductor generates a suffix
     xindex = idx_n + 768*idx_m
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
