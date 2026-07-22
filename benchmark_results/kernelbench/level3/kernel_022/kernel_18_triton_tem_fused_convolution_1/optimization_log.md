# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_18` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 30460 ns — **Best latency**: 21790 ns (**1.3979×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_warps_to_reduce_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_18`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_18_triton_tem_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.47
- **Registers/thread**: 255
- **Duration**: 30460 ns
- **SM Throughput**: 19.84 % | **Memory**: 32.22 % | **DRAM**: 0.96 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.94 inst/cycle | **No Eligible**: 76.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.002× | 0.963× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 30460 ns | 31620 ns | +1160 ns (+3.8%) |
| SM Throughput (%) | 19.84 % | 20.03 % | +0.19 pp |
| Memory Throughput (%) | 32.22 % | 32.50 % | +0.28 pp |
| DRAM Throughput (%) | 0.96 % | 0.91 % | -0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.47 | 0.47 | +0.00 |
| No Eligible (%) | 76.45 % | 76.29 % | -0.16 pp |
| IPC Active | 0.94 inst/cycle | 0.94 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_block_ptr | 1.246× | 1.408× | ✓ | **✓ best** |
| C | eviction_policies_streaming_loads | 1.000× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_block_p... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 30460 ns | 21630 ns | -8830 ns (-29.0%) | 30780 ns | +320 ns (+1.1%) |
| SM Throughput (%) | 19.84 % | 23.99 % | +4.15 pp | 20.17 % | +0.33 pp |
| Memory Throughput (%) | 32.22 % | 35.14 % | +2.92 pp | 32.76 % | +0.54 pp |
| DRAM Throughput (%) | 0.96 % | 1.33 % | +0.37 pp | 0.93 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.47 | 0.47 | +0.00 | 0.47 | +0.00 |
| No Eligible (%) | 76.45 % | 68.97 % | -7.48 pp | 76.14 % | -0.31 pp |
| IPC Active | 0.94 inst/cycle | 1.26 inst/cycle | +0.32 | 0.97 inst/cycle | +0.03 |

---

## Iteration 3

**Best so far**: 1.4082× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_async_pipeline_num_stages | 1.445× | 1.391× | ✓ | **✓ best** |
| C | tma_block_ptr_loads_to_reduce_address_gen_overhead | 1.242× | 1.406× | ✓ |  |
| D | tma_loads_tensor_descriptor | 1.296× | 1.483× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_async_p... | Δ | tma_block_ptr_loa... | Δ | tma_loads_tensor_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30460 ns | 21890 ns | -8570 ns (-28.1%) | 21660 ns | -8800 ns (-28.9%) | 20540 ns | -9920 ns (-32.6%) |
| SM Throughput (%) | 19.84 % | 22.68 % | +2.84 pp | 24.17 % | +4.33 pp | 23.76 % | +3.92 pp |
| Memory Throughput (%) | 32.22 % | 35.35 % | +3.13 pp | 35.51 % | +3.29 pp | 35.61 % | +3.39 pp |
| DRAM Throughput (%) | 0.96 % | 1.25 % | +0.29 pp | 1.33 % | +0.37 pp | 1.40 % | +0.44 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.47 | 0.47 | +0.00 | 0.47 | +0.00 | 0.47 | +0.00 |
| No Eligible (%) | 76.45 % | 70.38 % | -6.07 pp | 69.00 % | -7.45 pp | 68.90 % | -7.55 pp |
| IPC Active | 0.94 inst/cycle | 1.18 inst/cycle | +0.24 | 1.23 inst/cycle | +0.29 | 1.24 inst/cycle | +0.30 |


### Key Learnings

#### ✅ Plan C: `tma_block_ptr_loads_to_reduce_address_gen_overhead` (1.483×)

When a Triton matmul-style kernel uses complex scalar pointer arithmetic (manual index computation, broadcast expressions, masked loads) inside the K-loop, replacing these with tl.make_block_ptr descriptors and tl.advance() offloads address generation to Hopper's TMA hardware unit, freeing SM execution slots. This iteration applied the same transformation as item a9f2c841 but starting from an already-converted kernel (the 'before' state here was already using block_ptr), yielding a modest additional ~5% cycle reduction (32327→30729 cycles) and ~5% instruction reduction (4.07M→3.86M) primarily from refactoring stride variables and eliminating redundant scalar index rematerialization for the output store. The cumulative speedup from the original scalar-pointer baseline to this version is 1.483x.

> **Applicable when**: Triton kernel on Hopper (sm_90) uses manual scalar pointer arithmetic or broadcast index expressions inside a K-loop for matmul-style loads; NCU shows UncoalescedSharedAccess, high 'No Eligible' scheduler percentage (>65%), low IPC active (<1.3), and Issue Slots Busy <35%. Even after partial block_ptr conversion, replacing scalar output stores with block_ptr stores and named stride variables can yield additional small instruction-count reductions.

> **Avoid when**: Non-Hopper architectures (pre-sm_90) where TMA hardware is unavailable; kernels where the tensor layout cannot be expressed as a 2D block pointer (e.g., truly irregular/sparse access patterns); cases where boundary_check overhead dominates (K dimension is large and always divisible by BLOCK_K). If the kernel is already fully converted to block_ptr for all loads and stores, further gains from this direction are exhausted.

---

## Iteration 4

**Best so far**: 1.4082× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_reduce_shared_memory_bank_conflicts | 1.449× | 1.398× | ✓ | **✓ best** |
| B | cap_registers_with_maxnreg_and_block_ptr_output_store | 1.432× | 1.398× | ✓ |  |
| C | autotune_num_warps_and_maxnreg_to_resolve_bank_conflicts_and_occupancy | 1.374× | 1.393× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | cap_registers_wit... | Δ | autotune_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30460 ns | 21790 ns | -8670 ns (-28.5%) | 21790 ns | -8670 ns (-28.5%) | 21860 ns | -8600 ns (-28.2%) |
| SM Throughput (%) | 19.84 % | 22.66 % | +2.82 pp | 22.45 % | +2.61 pp | 22.69 % | +2.85 pp |
| Memory Throughput (%) | 32.22 % | 35.43 % | +3.21 pp | 35.08 % | +2.86 pp | 35.47 % | +3.25 pp |
| DRAM Throughput (%) | 0.96 % | 1.26 % | +0.30 pp | 1.26 % | +0.30 pp | 1.25 % | +0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.47 | 0.47 | +0.00 | 0.47 | +0.00 | 0.47 | +0.00 |
| No Eligible (%) | 76.45 % | 70.46 % | -5.99 pp | 70.33 % | -6.12 pp | 70.58 % | -5.87 pp |
| IPC Active | 0.94 inst/cycle | 1.16 inst/cycle | +0.22 | 1.18 inst/cycle | +0.24 | 1.16 inst/cycle | +0.22 |

---

## Iteration 5

**Best so far**: 1.4082× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_block_k_to_reduce_smem_per_stage_and_expand_search_space | 1.438× | 1.390× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_block_k_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 30460 ns | 21920 ns | -8540 ns (-28.0%) |
| SM Throughput (%) | 19.84 % | 22.49 % | +2.65 pp |
| Memory Throughput (%) | 32.22 % | 35.17 % | +2.95 pp |
| DRAM Throughput (%) | 0.96 % | 1.25 % | +0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.24 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.47 | 0.47 | +0.00 |
| No Eligible (%) | 76.45 % | 70.55 % | -5.90 pp |
| IPC Active | 0.94 inst/cycle | 1.17 inst/cycle | +0.23 |

---

## Cumulative Progress

```
Baseline:  30460 ns  (1.000×)
Iter 2     21630 ns  (1.408×)  — tma_loads_block_ptr

Final best: 21790 ns  (1.3979×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_18(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -12,7 +24,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -44,40 +56,26 @@

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
 
+    # Create TMA block pointers for A and B
+    # A: shape (M=7840, K=40), strides (40, 1), row-major -> order=(1, 0)
+    a_block_ptr = tl.make_block_ptr(
+        base=A, shape=(M, K), strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
+    # B: accessed as (K=40, N=240), strides (stride_bk=1, stride_bn=40) -> order=(0, 1)
+    b_block_ptr = tl.make_block_ptr(
+        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
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
-        xindex = idx_n + 40*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 240*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 40*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
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
@@ -88,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 240*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
