# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_add_addmm_native_layer_norm_t_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16520000 ns — **Best latency**: 16500000 ns (**1.0012×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`eviction_policies_streaming_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_add_addmm_native_layer_norm_t_view_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_11_triton_tem_fused_add_addmm_nat/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12288,1,1), Waves/SM: 46.55
- **Registers/thread**: 255
- **Duration**: 16520000 ns
- **SM Throughput**: 41.57 % | **Memory**: 60.68 % | **DRAM**: 1.75 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.001× | ✓ |  |
| B | tma_loads_with_autotuned_tile_sizes | 0.956× | 0.169× | ✓ |  |
| C | eviction_policies_streaming_loads | 1.001× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16520000 ns | 16510000 ns | -10000 ns (-0.1%) | 97530000 ns | +81010000 ns (+490.4%) | 16500000 ns | -20000 ns (-0.1%) |
| SM Throughput (%) | 41.57 % | 41.54 % | -0.03 pp | 8.56 % | -33.01 pp | 41.46 % | -0.11 pp |
| Memory Throughput (%) | 60.68 % | 60.63 % | -0.05 pp | 17.11 % | -43.57 pp | 60.52 % | -0.16 pp |
| DRAM Throughput (%) | 1.75 % | 1.75 % | +0.00 pp | 12.31 % | +10.56 pp | 1.57 % | -0.18 pp |
| Achieved Occupancy (%) | 12.46 % | 12.46 % | +0.00 pp | 6.25 % | -6.21 pp | 12.46 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 | 93.09 | +46.54 | 46.55 | +0.00 |
| No Eligible (%) | 58.11 % | 58.15 % | +0.04 pp | 91.35 % | +33.24 pp | 58.16 % | +0.05 pp |
| IPC Active | 1.67 inst/cycle | 1.67 inst/cycle | +0.00 | 0.34 inst/cycle | -1.33 | 1.68 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0012× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | revert_block_ptr_to_scalar_loads_to_eliminate_smem_explosion | 0.916× | 0.211× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_block_ptr_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16520000 ns | 78300000 ns | +61780000 ns (+374.0%) |
| SM Throughput (%) | 41.57 % | 10.60 % | -30.97 pp |
| Memory Throughput (%) | 60.68 % | 22.72 % | -37.96 pp |
| DRAM Throughput (%) | 1.75 % | 18.58 % | +16.83 pp |
| Achieved Occupancy (%) | 12.46 % | 12.47 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 |
| No Eligible (%) | 58.11 % | 89.33 % | +31.22 pp |
| IPC Active | 1.67 inst/cycle | 0.43 inst/cycle | -1.24 |


### Key Learnings

#### ❌ Plan C: `revert_block_ptr_to_scalar_loads_to_eliminate_smem_explosion` (0.211×)

The revert from tl.make_block_ptr to scalar pointer arithmetic preserved occupancy (12.5%) and shared memory (65 KB) but caused a 4.75x duration regression (16ms → 78ms). The stack size tripled (1024 → 3104 bytes), indicating the new pointer arithmetic introduced spills or complex address computation. The dominant new bottleneck is CPIStall from L1TEX: 72.3% of warp cycles are stalled waiting on L1TEX loads, with Warp Cycles Per Issued Instruction rising from 4.76 to 18.68 and eligible warps dropping from 0.73 to 0.12 per scheduler. The b_ptrs indexing formula (offs_n[None,:]*K + offs_k[:,None]) was incorrect for the transposed B matrix layout, generating non-coalesced or strided accesses that thrash L1TEX and cause massive memory latency stalls.

> **Applicable when**: Never: this specific rewrite caused regression. The original scalar load pattern with explicit broadcast indexing was correct and fast.

> **Avoid when**: Always avoid this rewrite pattern. When reverting from tl.make_block_ptr to scalar loads, verify that pointer stride arithmetic exactly matches the original kernel's memory layout. Incorrect b_ptrs stride (using offs_n*K instead of offs_n*1 for a column-major or transposed matrix) causes non-coalesced L1TEX accesses, tripling warp stall cycles and causing CPIStall to dominate at 72%+ of execution time. The tripled stack size (1024→3104) is a diagnostic signal that address computation spilled to local memory.

---

## Iteration 3

**Best so far**: 1.0012× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_pointer_arithmetic_correctness | 0.916× | 0.211× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_poin... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16520000 ns | 78160000 ns | +61640000 ns (+373.1%) |
| SM Throughput (%) | 41.57 % | 10.60 % | -30.97 pp |
| Memory Throughput (%) | 60.68 % | 22.71 % | -37.97 pp |
| DRAM Throughput (%) | 1.75 % | 18.61 % | +16.86 pp |
| Achieved Occupancy (%) | 12.46 % | 12.47 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 46.55 | 46.55 | +0.00 |
| No Eligible (%) | 58.11 % | 89.33 % | +31.22 pp |
| IPC Active | 1.67 inst/cycle | 0.43 inst/cycle | -1.24 |


### Key Learnings

#### ❌ Plan A: `fix_b_matrix_pointer_arithmetic_correctness` (0.211×)

Replacing pre-loop pointer variables with per-iteration explicit index recomputation (offs_k_cur = k_idx*BLOCK_K + tl.arange(0,BLOCK_K)) produced no measurable improvement: duration remained 78.16ms vs 78.3ms before, stack size stayed at 3104 bytes, eligible warps stayed at 0.12, and CPIStall from L1TEX remained at 72%+ of warp cycles. The profiling metrics are essentially identical before and after, indicating the root cause of the regression is not the pointer increment direction but something deeper — likely the overall scalar load pattern itself or the kernel structure inherited from the prior broken rewrite. When all key metrics (duration, stack size, occupancy, scheduler eligibility, warp stall cycles) are unchanged after a structural pointer arithmetic fix, the bottleneck is not addressable by index recomputation alone.

> **Applicable when**: Never: this rewrite produced no improvement. The underlying kernel state (stack=3104, CPIStall=72%, eligible warps=0.12) is unchanged by this structural change.

> **Avoid when**: When the kernel is already in a degraded state with stack size 3104 bytes, CPIStall at 72%+ of warp cycles, and eligible warps at 0.12 per scheduler — switching from pre-loop pointer increments to per-iteration index recomputation does not recover performance. The bottleneck is not the pointer increment direction. Do not apply this as a standalone fix when all profiling metrics remain identical after the change.

---

## Cumulative Progress

```
Baseline:  16520000 ns  (1.000×)
Iter 1     16500000 ns  (1.001×)  — eviction_policies_streaming_loads

Final best: 16500000 ns  (1.0012×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4,  num_stages=1),
+        triton.Config({}, num_warps=8,  num_stages=1),
+        triton.Config({}, num_warps=16, num_stages=1),
+        triton.Config({}, num_warps=4,  num_stages=2),
+        triton.Config({}, num_warps=8,  num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_add_addmm_native_layer_norm_t_view_11(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -12,7 +23,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -22,13 +33,10 @@

     M = 65536
     N = 3072
     K = 768
+
     if M * N == 0:
         # early exit due to zero-size input(s)
         return
-    stride_am = 768
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 768
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
@@ -44,50 +52,28 @@

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
+    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # shape [BLOCK_M]
+    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # shape [BLOCK_N]
+    offs_k = tl.arange(0, BLOCK_K)                      # shape [BLOCK_K]
+
+    # A: shape (M, K), row-major strides (K, 1)
+    a_ptrs = arg_A + offs_m[:, None] * K + offs_k[None, :]  # [BLOCK_M, BLOCK_K]
+    # B: physical shape (N=3072, K=768), strides (768, 1)
+    # Accessed transposed as (K=768, N=3072): B[k, n] = arg_B + n*K + k*1
+    b_ptrs = arg_B + offs_n[None, :] * K + offs_k[:, None] * 1  # [BLOCK_K, BLOCK_N]
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        # EVEN_K=True: K=768 divisible by BLOCK_K=64 — no mask needed
+        a = tl.load(a_ptrs)   # [BLOCK_M, BLOCK_K]
+        b = tl.load(b_ptrs)   # [BLOCK_K, BLOCK_N]
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        a_ptrs += BLOCK_K     # advance K-dim of A (stride=1 per element)
+        b_ptrs += BLOCK_K     # advance K-dim of B (stride=1 per element)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 768*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 3072*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 3072*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    # Store: M=65536 div by 128, N=3072 div by 128 — no mask needed
+    out_ptrs = out_ptr0 + offs_m[:, None] * N + offs_n[None, :]  # [BLOCK_M, BLOCK_N]
+    tl.store(out_ptrs, acc)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
