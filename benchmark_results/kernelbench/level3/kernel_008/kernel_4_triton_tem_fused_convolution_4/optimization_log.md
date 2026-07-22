# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1700000 ns — **Best latency**: 1700000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_008/kernel_4_triton_tem_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 9.90
- **Registers/thread**: 168
- **Duration**: 1700000 ns
- **SM Throughput**: 40.46 % | **Memory**: 91.54 % | **DRAM**: 12.95 %
- **Achieved Occupancy**: 18.41 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 58.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | — | 1.000× | ✓ | **✓ best** |
| D | add_autotune_with_larger_tiles_and_pipelining | — | 0.944× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_with... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1700000 ns | +0 ns (+0.0%) | 1800000 ns | +100000 ns (+5.9%) |
| SM Throughput (%) | 40.46 % | 40.49 % | +0.03 pp | 37.73 % | -2.73 pp |
| Memory Throughput (%) | 91.54 % | 91.60 % | +0.06 pp | 85.99 % | -5.55 pp |
| DRAM Throughput (%) | 12.95 % | 12.96 % | +0.01 pp | 12.23 % | -0.72 pp |
| Achieved Occupancy (%) | 18.41 % | 18.41 % | +0.00 pp | 18.38 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 58.96 % | +0.00 pp | 61.57 % | +2.61 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 | 1.54 inst/cycle | -0.10 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_to_eliminate_shared_memory_bank_conflicts | — | 0.994× | ✓ |  |
| B | tma_loads_fix_b_matrix_access_pattern | — | 1.000× | ✓ |  |
| C | fix_b_matrix_transposed_load_to_eliminate_bank_conflicts | — | 0.939× | ✓ |  |
| D | tma_loads_via_block_ptr_fix_bank_conflicts | — | 0.977× | ✓ |  |
| E | add_triton_autotune | — | 0.934× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_fix_b_m... | Δ | fix_b_matrix_tran... | Δ | tma_loads_via_blo... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1710000 ns | +10000 ns (+0.6%) | 1700000 ns | +0 ns (+0.0%) | 1810000 ns | +110000 ns (+6.5%) | 1740000 ns | +40000 ns (+2.4%) | 1820000 ns | +120000 ns (+7.1%) |
| SM Throughput (%) | 40.46 % | 40.39 % | -0.07 pp | 40.54 % | +0.08 pp | 37.85 % | -2.61 pp | 39.50 % | -0.96 pp | 37.76 % | -2.70 pp |
| Memory Throughput (%) | 91.54 % | 91.37 % | -0.17 pp | 91.56 % | +0.02 pp | 86.27 % | -5.27 pp | 89.66 % | -1.88 pp | 86.07 % | -5.47 pp |
| DRAM Throughput (%) | 12.95 % | 12.94 % | -0.01 pp | 12.96 % | +0.01 pp | 12.22 % | -0.73 pp | 12.71 % | -0.24 pp | 12.14 % | -0.81 pp |
| Achieved Occupancy (%) | 18.41 % | 18.41 % | +0.00 pp | 18.41 % | +0.00 pp | 18.38 % | -0.03 pp | 18.42 % | +0.01 pp | 18.38 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 58.98 % | +0.02 pp | 58.88 % | -0.08 pp | 61.55 % | +2.59 pp | 59.88 % | +0.92 pp | 61.57 % | +2.61 pp |
| IPC Active | 1.64 inst/cycle | 1.64 inst/cycle | +0.00 | 1.64 inst/cycle | +0.00 | 1.54 inst/cycle | -0.10 | 1.60 inst/cycle | -0.04 | 1.54 inst/cycle | -0.10 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_to_eliminate_bank_conflicts | — | 0.971× | ✓ |  |
| B | fix_b_block_ptr_order_to_eliminate_smem_bank_conflicts | — | 0.833× | ✓ |  |
| C | use_block_ptr_for_automatic_shared_memory_swizzling | — | 0.994× | ✓ |  |
| D | transpose_b_tile_in_shared_memory_to_eliminate_bank_conflicts | — | 0.939× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | fix_b_block_ptr_o... | Δ | use_block_ptr_for... | Δ | transpose_b_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1700000 ns | 1750000 ns | +50000 ns (+2.9%) | 2040000 ns | +340000 ns (+20.0%) | 1710000 ns | +10000 ns (+0.6%) | 1810000 ns | +110000 ns (+6.5%) |
| SM Throughput (%) | 40.46 % | 39.56 % | -0.90 pp | 33.38 % | -7.08 pp | 40.54 % | +0.08 pp | 37.82 % | -2.64 pp |
| Memory Throughput (%) | 91.54 % | 89.08 % | -2.46 pp | 76.00 % | -15.54 pp | 91.55 % | +0.01 pp | 86.19 % | -5.35 pp |
| DRAM Throughput (%) | 12.95 % | 12.58 % | -0.37 pp | 10.86 % | -2.09 pp | 12.94 % | -0.01 pp | 12.21 % | -0.74 pp |
| Achieved Occupancy (%) | 18.41 % | 18.40 % | -0.01 pp | 12.46 % | -5.95 pp | 18.41 % | +0.00 pp | 18.39 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 244.00 register/thread | +76 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 9.90 | 9.90 | +0.00 | 14.85 | +4.95 | 9.90 | +0.00 | 9.90 | +0.00 |
| No Eligible (%) | 58.96 % | 59.80 % | +0.84 pp | 66.06 % | +7.10 pp | 58.88 % | -0.08 pp | 61.55 % | +2.59 pp |
| IPC Active | 1.64 inst/cycle | 1.61 inst/cycle | -0.03 | 1.36 inst/cycle | -0.28 | 1.64 inst/cycle | +0.00 | 1.54 inst/cycle | -0.10 |


### Key Learnings

#### ❌ Plan B: `fix_b_block_ptr_order_to_eliminate_smem_bank_conflicts` (0.833×)

Changing the `order` parameter of `tl.make_block_ptr` for matrix B from (0,1) to (1,0) while simultaneously switching matrix A from a block-pointer load to a manual index-based load caused registers per thread to jump from 168 to 244. This register explosion reduced theoretical occupancy from 18.75% to 12.5% (Block Limit Registers dropped from 3 to 2 blocks/SM), increasing kernel duration from 1.7ms to 2.04ms despite slightly reducing shared memory usage. The bank conflict counts remained essentially unchanged (210M+ excessive wavefronts), confirming the order change alone did not resolve the root cause while the manual A-load rewrite introduced significant register pressure.

> **Applicable when**: Never: this specific combination of changing B order=(1,0) while replacing A block-pointer with manual index loads is strictly worse.

> **Avoid when**: Any kernel where registers per thread are already high (>150). Replacing a tl.make_block_ptr load with manual index arithmetic (xindex = offset + stride * idx) dramatically increases register pressure because the compiler must materialize all index expressions as live registers. Even if the intent is to fix shared memory bank conflicts, the register explosion from manual loads outweighs any benefit from the order change, reducing occupancy and increasing runtime. The bank conflict count was also unchanged, confirming the order change alone did not fix the underlying layout mismatch.

---

## Cumulative Progress

```
Baseline:  1700000 ns  (1.000×)

Final best: 1700000 ns  (1.0000×)
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
 def triton_tem_fused_convolution_4(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -44,37 +45,32 @@

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
 
+    # Construct TMA block pointers before the k-loop
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),  # (480, 1)
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)  # K-dim (last) is contiguous
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),  # (1, 480)
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)  # K-dim (first) is contiguous
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 480*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +84,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
