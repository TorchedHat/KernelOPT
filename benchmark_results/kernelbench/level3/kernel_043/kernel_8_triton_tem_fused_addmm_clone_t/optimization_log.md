# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_clone_t_transpose_view_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4230000 ns — **Best latency**: 3800000 ns (**1.1132×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_tl_trans_bank_conflicts_via_dot_trans_b`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_clone_t_transpose_view_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_8_triton_tem_fused_addmm_clone_t/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 11.64
- **Registers/thread**: 255
- **Duration**: 4230000 ns
- **SM Throughput**: 40.60 % | **Memory**: 59.26 % | **DRAM**: 2.17 %
- **Achieved Occupancy**: 12.35 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4230000 ns | 4230000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 40.60 % | 40.54 % | -0.06 pp |
| Memory Throughput (%) | 59.26 % | 59.17 % | -0.09 pp |
| DRAM Throughput (%) | 2.17 % | 2.18 % | +0.01 pp |
| Achieved Occupancy (%) | 12.35 % | 12.36 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 11.64 | +0.00 |
| No Eligible (%) | 58.34 % | 58.39 % | +0.05 pp |
| IPC Active | 1.67 inst/cycle | 1.67 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_b_matrix_load_correctness_and_autotune_tile_sizes | 1.000× | 0.998× | ✓ |  |
| E | add_triton_autotune | 1.013× | 0.964× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4230000 ns | 4240000 ns | +10000 ns (+0.2%) | 4390000 ns | +160000 ns (+3.8%) |
| SM Throughput (%) | 40.60 % | 40.58 % | -0.02 pp | 39.13 % | -1.47 pp |
| Memory Throughput (%) | 59.26 % | 59.24 % | -0.02 pp | 60.98 % | +1.72 pp |
| DRAM Throughput (%) | 2.17 % | 2.17 % | +0.00 pp | 2.20 % | +0.03 pp |
| Achieved Occupancy (%) | 12.35 % | 12.34 % | -0.01 pp | 12.41 % | +0.06 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 11.64 | +0.00 | 11.64 | +0.00 |
| No Eligible (%) | 58.34 % | 58.28 % | -0.06 pp | 59.92 % | +1.58 pp |
| IPC Active | 1.67 inst/cycle | 1.67 inst/cycle | +0.00 | 1.62 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_coalesced_b_via_block_ptr | 1.113× | 1.113× | ✓ | **✓ best** |
| D | fix_b_matrix_load_address_computation_correctly | 1.004× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_coalesc... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4230000 ns | 3800000 ns | -430000 ns (-10.2%) | 4390000 ns | +160000 ns (+3.8%) |
| SM Throughput (%) | 40.60 % | 45.35 % | +4.75 pp | 39.38 % | -1.22 pp |
| Memory Throughput (%) | 59.26 % | 63.22 % | +3.96 pp | 61.40 % | +2.14 pp |
| DRAM Throughput (%) | 2.17 % | 2.33 % | +0.16 pp | 2.20 % | +0.03 pp |
| Achieved Occupancy (%) | 12.35 % | 12.38 % | +0.03 pp | 12.32 % | -0.03 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 11.64 | +0.00 | 11.64 | +0.00 |
| No Eligible (%) | 58.34 % | 53.32 % | -5.02 pp | 59.81 % | +1.47 pp |
| IPC Active | 1.67 inst/cycle | 1.87 inst/cycle | +0.20 | 1.61 inst/cycle | -0.06 |


### Key Learnings

#### ✅ Plan B: `tma_loads_coalesced_b_via_block_ptr` (1.113×)

When a GEMM kernel loads the B matrix with a stride-N access pattern (only 1/32 bytes per sector utilized, triggering MemoryCacheAccessPattern and UncoalescedSharedAccess NCU rules), reloading B as [BLOCK_N, BLOCK_K] (the contiguous layout) and then calling tl.trans() to get [BLOCK_K, BLOCK_N] for tl.dot eliminates the strided access. This change improved L1/TEX hit rate from 7.1% to 35.2%, reduced dynamic shared memory per block from 65 KB to 33 KB, cut elapsed cycles by ~10%, and improved compute throughput from 40.6% to 45.4%. Simultaneously reducing BLOCK_K from 32 to 16 further reduces register pressure and K-loop body live values.

> **Applicable when**: Triton GEMM kernel where the B matrix has stride_bk=1 (K is the fast/column dimension) and stride_bn=N (N is the slow/row dimension), but the load pattern iterates over K in the outer index and N in the inner index, causing stride-N uncoalesced access. NCU signals: MemoryCacheAccessPattern reporting only 1/32 bytes per sector utilized for local loads/stores, UncoalescedSharedAccess with >50% excessive wavefronts, SharedMemoryConflicts with >5x average bank conflict way, and L1/TEX Hit Rate below 10%.

> **Avoid when**: B matrix is already loaded in a coalesced pattern (L1/TEX hit rate >50%, bytes-per-sector utilization near 32/32). Also avoid if the B matrix layout has stride_bn=1 (N is the fast dimension), as the transpose would then create the uncoalesced access instead of fixing it. Do not apply if tl.trans causes compiler issues with the specific BLOCK_N/BLOCK_K shapes used.

---

## Iteration 4

**Best so far**: 1.1132× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_tl_trans_bank_conflicts_via_dot_trans_b | 1.113× | 1.113× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4230000 ns | 3800000 ns | -430000 ns (-10.2%) |
| SM Throughput (%) | 40.60 % | 45.32 % | +4.72 pp |
| Memory Throughput (%) | 59.26 % | 63.18 % | +3.92 pp |
| DRAM Throughput (%) | 2.17 % | 2.34 % | +0.17 pp |
| Achieved Occupancy (%) | 12.35 % | 12.37 % | +0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 11.64 | +0.00 |
| No Eligible (%) | 58.34 % | 53.28 % | -5.06 pp |
| IPC Active | 1.67 inst/cycle | 1.87 inst/cycle | +0.20 |

---

## Iteration 5

**Best so far**: 1.1132× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_a_with_software_pipelining_num_stages | 1.101× | 1.102× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_a_with_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4230000 ns | 3840000 ns | -390000 ns (-9.2%) |
| SM Throughput (%) | 40.60 % | 44.82 % | +4.22 pp |
| Memory Throughput (%) | 59.26 % | 62.07 % | +2.81 pp |
| DRAM Throughput (%) | 2.17 % | 2.26 % | +0.09 pp |
| Achieved Occupancy (%) | 12.35 % | 12.34 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 11.64 | +0.00 |
| No Eligible (%) | 58.34 % | 53.88 % | -4.46 pp |
| IPC Active | 1.67 inst/cycle | 1.84 inst/cycle | +0.17 |

---

## Cumulative Progress

```
Baseline:  4230000 ns  (1.000×)
Iter 3     3800000 ns  (1.113×)  — tma_loads_coalesced_b_via_block_ptr

Final best: 3800000 ns  (1.1132×)
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
 def triton_tem_fused_addmm_clone_t_transpose_view_8(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -12,7 +13,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,23 +59,22 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        k_start = k_idx * BLOCK_K
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        # A load: A[m, k] = A_ptr + m*stride_am + k*stride_ak = A_ptr + m*768 + k*1
+        a_offs_m = offs_a_m[:, None]           # shape [BLOCK_M, 1]
+        a_offs_k = offs_k[None, :] + k_start  # shape [1, BLOCK_K]
+        a = tl.load(A + a_offs_m * stride_am + a_offs_k * stride_ak)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 768*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 768*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B load: load as [BLOCK_N, BLOCK_K] — N rows, each row has BLOCK_K contiguous K elements
+        # B[n, k] = B_ptr + n*stride_bn + k*stride_bk = B_ptr + n*768 + k*1
+        # Each warp accesses BLOCK_K=16 contiguous floats (stride_bk=1) — fully coalesced!
+        b_offs_n_col = offs_b_n[:, None]           # [BLOCK_N, 1]
+        b_offs_k_row = offs_k[None, :] + k_start  # [1, BLOCK_K]
+        b_T = tl.load(B + b_offs_n_col * stride_bn + b_offs_k_row * stride_bk)  # [BLOCK_N, BLOCK_K]
+        b = tl.trans(b_T)  # transpose to [BLOCK_K, BLOCK_N] for tl.dot
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +88,6 @@

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
