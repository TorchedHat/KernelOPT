# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4010000 ns — **Best latency**: 4010000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_3_triton_tem_fused_addmm_t_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 4010000 ns
- **SM Throughput**: 37.73 % | **Memory**: 86.12 % | **DRAM**: 3.14 %
- **Achieved Occupancy**: 12.25 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | add_autotune_fix_register_pressure_and_bank_conflicts | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_fix_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4010000 ns | 4020000 ns | +10000 ns (+0.2%) |
| SM Throughput (%) | 37.73 % | 37.70 % | -0.03 pp |
| Memory Throughput (%) | 86.12 % | 86.05 % | -0.07 pp |
| DRAM Throughput (%) | 3.14 % | 3.14 % | +0.00 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 |
| No Eligible (%) | 61.24 % | 61.24 % | +0.00 pp |
| IPC Active | 1.55 inst/cycle | 1.55 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_fix_b_matrix_coalescing | 0.989× | 0.990× | ✓ |  |
| C | fix_b_matrix_load_coalescing_via_transposed_tile | 1.001× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_fix_b_m... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4010000 ns | 4050000 ns | +40000 ns (+1.0%) | 4010000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 37.73 % | 37.24 % | -0.49 pp | 37.80 % | +0.07 pp |
| Memory Throughput (%) | 86.12 % | 84.50 % | -1.62 pp | 86.28 % | +0.16 pp |
| DRAM Throughput (%) | 3.14 % | 3.08 % | -0.06 pp | 3.14 % | +0.00 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 61.24 % | 61.77 % | +0.53 pp | 61.24 % | +0.00 pp |
| IPC Active | 1.55 inst/cycle | 1.53 inst/cycle | -0.02 | 1.55 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_eliminate_trans_spill | 0.987× | 0.988× | ✓ |  |
| C | eliminate_tl_trans_shared_memory_bank_conflicts | 0.987× | 0.988× | ✓ |  |
| D | eliminate_b_transpose_bank_conflicts_via_native_block_ptr_order | 0.987× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | eliminate_tl_tran... | Δ | eliminate_b_trans... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4010000 ns | 4060000 ns | +50000 ns (+1.2%) | 4060000 ns | +50000 ns (+1.2%) | 4060000 ns | +50000 ns (+1.2%) |
| SM Throughput (%) | 37.73 % | 37.12 % | -0.61 pp | 37.05 % | -0.68 pp | 37.14 % | -0.59 pp |
| Memory Throughput (%) | 86.12 % | 84.22 % | -1.90 pp | 84.08 % | -2.04 pp | 84.27 % | -1.85 pp |
| DRAM Throughput (%) | 3.14 % | 3.08 % | -0.06 pp | 3.08 % | -0.06 pp | 3.08 % | -0.06 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp | 12.24 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 61.24 % | 61.80 % | +0.56 pp | 61.81 % | +0.57 pp | 61.82 % | +0.58 pp |
| IPC Active | 1.55 inst/cycle | 1.53 inst/cycle | -0.02 | 1.53 inst/cycle | -0.02 | 1.53 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores_ieee_precision_via_input_precision_arg | 0.988× | 0.988× | ✓ |  |
| B | fix_b_block_ptr_order_for_coalesced_loads | 0.987× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4010000 ns | 4060000 ns | +50000 ns (+1.2%) | 4060000 ns | +50000 ns (+1.2%) |
| SM Throughput (%) | 37.73 % | 37.04 % | -0.69 pp | 37.14 % | -0.59 pp |
| Memory Throughput (%) | 86.12 % | 84.06 % | -2.06 pp | 84.27 % | -1.85 pp |
| DRAM Throughput (%) | 3.14 % | 3.08 % | -0.06 pp | 3.08 % | -0.06 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 61.24 % | 61.80 % | +0.56 pp | 61.81 % | +0.57 pp |
| IPC Active | 1.55 inst/cycle | 1.53 inst/cycle | -0.02 | 1.53 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  4010000 ns  (1.000×)

Final best: 4010000 ns  (1.0000×)
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
 def triton_tem_fused_addmm_t_4(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -44,37 +45,36 @@

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
+    # Create block pointers for A and B using TMA-friendly descriptors
+    # A: shape (M, K), strides (stride_am=16384, stride_ak=1), row-major
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B physical layout: B[n, k] at offset n*16384 + k*1
+    # We need B^T[k, n] for tl.dot(A[M,K], B^T[K,N])
+    # Declare as shape=(K, N) with strides=(stride_bk=1, stride_bn=16384):
+    # element (k_local, n_local) -> k_local*1 + n_local*16384 = B[n_local, k_local] = B^T[k_local, n_local]
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)
+    )
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 16384*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 8192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
+    for _ in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))  # [BLOCK_M, BLOCK_K]
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))  # [BLOCK_K, BLOCK_N] — no tl.trans needed
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
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
