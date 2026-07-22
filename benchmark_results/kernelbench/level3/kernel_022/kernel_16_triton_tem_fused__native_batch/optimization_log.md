# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21060 ns — **Best latency**: 26140 ns (**0.8057×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_16_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 21060 ns
- **SM Throughput**: 18.64 % | **Memory**: 70.35 % | **DRAM**: 4.42 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.89 inst/cycle | **No Eligible**: 77.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_streaming_loads | 1.007× | 0.996× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 21060 ns | 21150 ns | +90 ns (+0.4%) |
| SM Throughput (%) | 18.64 % | 18.76 % | +0.12 pp |
| Memory Throughput (%) | 70.35 % | 70.73 % | +0.38 pp |
| DRAM Throughput (%) | 4.42 % | 4.40 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.49 % | 77.48 % | -0.01 pp |
| IPC Active | 0.89 inst/cycle | 0.89 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_padding_to_eliminate_bank_conflicts | 0.998× | 0.991× | ✓ |  |
| B | tma_loads | 1.006× | 1.008× | ✓ |  |
| C | fix_b_matrix_load_coalescing_eliminate_bank_conflicts | 0.994× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ | tma_loads | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21060 ns | 21250 ns | +190 ns (+0.9%) | 20900 ns | -160 ns (-0.8%) | 21180 ns | +120 ns (+0.6%) |
| SM Throughput (%) | 18.64 % | 18.66 % | +0.02 pp | 18.59 % | -0.05 pp | 18.62 % | -0.02 pp |
| Memory Throughput (%) | 70.35 % | 70.53 % | +0.18 pp | 71.50 % | +1.15 pp | 70.20 % | -0.15 pp |
| DRAM Throughput (%) | 4.42 % | 4.38 % | -0.04 pp | 4.45 % | +0.03 pp | 4.40 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 77.49 % | 77.62 % | +0.13 pp | 77.64 % | +0.15 pp | 77.55 % | +0.06 pp |
| IPC Active | 0.89 inst/cycle | 0.89 inst/cycle | +0.00 | 0.88 inst/cycle | -0.01 | 0.89 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_with_correct_dynamic_grid | 1.009× | 0.809× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_with... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 21060 ns | 26020 ns | +4960 ns (+23.6%) |
| SM Throughput (%) | 18.64 % | 14.24 % | -4.40 pp |
| Memory Throughput (%) | 70.35 % | 32.82 % | -37.53 pp |
| DRAM Throughput (%) | 4.42 % | 3.59 % | -0.83 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 168.00 register/thread | +40 |
| Waves/SM | 0.23 | 0.16 | -0.07 |
| No Eligible (%) | 77.49 % | 63.71 % | -13.78 pp |
| IPC Active | 0.89 inst/cycle | 0.72 inst/cycle | -0.17 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_load_eliminate_transpose_and_bank_conflicts | 1.016× | 0.806× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 21060 ns | 26140 ns | +5080 ns (+24.1%) |
| SM Throughput (%) | 18.64 % | 14.34 % | -4.30 pp |
| Memory Throughput (%) | 70.35 % | 32.98 % | -37.37 pp |
| DRAM Throughput (%) | 4.42 % | 3.57 % | -0.85 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 168.00 register/thread | +40 |
| Waves/SM | 0.23 | 0.16 | -0.07 |
| No Eligible (%) | 77.49 % | 63.64 % | -13.85 pp |
| IPC Active | 0.89 inst/cycle | 0.72 inst/cycle | -0.17 |


### Key Learnings

#### ❌ Plan B: `fix_b_matrix_load_eliminate_transpose_and_bank_conflicts` (0.806×)

Rewriting the B-matrix block pointer to load [BLOCK_K, BLOCK_N] directly (avoiding tl.trans()) produced no measurable improvement: duration increased from 26.02us to 26.14us (0.81x), all occupancy, register, and memory metrics remained identical, and the SharedMemoryConflicts rule still reported 7.9-way bank conflicts with 69.31% conflict wavefronts. The profiling data shows the bank conflicts and uncoalesced shared accesses are not caused by the transpose operation itself — they persist unchanged after the transpose is removed. When all key metrics (registers/thread, occupancy, bank conflict counts, instruction count) are statistically unchanged before and after, the structural change did not address the true root cause of the shared memory access pattern.

> **Applicable when**: Never — this is a negative result. Do not apply this transformation expecting bank conflict reduction when the SharedMemoryConflicts and UncoalescedSharedAccess rules persist unchanged after the transpose is removed.

> **Avoid when**: When the kernel already has 168 registers/thread, 3.12% achieved occupancy, and 7.9-way shared memory bank conflicts that are not caused by a tl.trans() call — eliminating the transpose does not change any profiling metric and produces no speedup. The bank conflicts originate elsewhere in the kernel (e.g., the A-matrix or output store access patterns, or the register-pressure-driven spill/fill traffic).

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  21060 ns  (1.000×)

Final best: 26140 ns  (0.8057×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -44,40 +58,40 @@

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
 
+    # Create block pointers ONCE before the k-loop using TMA-style 2D descriptors
+    # A is [M=7840, K=144], row-major strides [144, 1]
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # Reinterpret B storage [N=40, K=144] as [K=144, N=40] to load [BLOCK_K, BLOCK_N] directly
+    # strides: (stride_bk=1, stride_bn=144) — K is dim-0 (fast inner), N is dim-1
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 40*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
+        # Load A tile [BLOCK_M, BLOCK_K] — TMA handles boundary with zero-padding
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        # Load B tile [BLOCK_K, BLOCK_N] directly — no transpose needed
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
+        # Advance block pointers along K dimension
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,9 +102,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 40*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
