# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_view_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3580 ns — **Best latency**: 2780 ns (**1.2878×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`outer_product_accumulation_with_hardcoded_static_range_and_wider_block_n`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_view_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_7_triton_tem_fused_addmm_t_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.07
- **Registers/thread**: 44
- **Duration**: 3580 ns
- **SM Throughput**: 4.14 % | **Memory**: 17.59 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.06 %
- **IPC Active**: 0.47 inst/cycle | **No Eligible**: 88.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.008× | 1.000× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_fix_block_k_for_tiny_k_gemm | 0.543× | 0.372× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 3580 ns | +0 ns (+0.0%) | 9630 ns | +6050 ns (+169.0%) |
| SM Throughput (%) | 4.14 % | 3.88 % | -0.26 pp | 13.73 % | +9.59 pp |
| Memory Throughput (%) | 17.59 % | 16.46 % | -1.13 pp | 22.89 % | +5.30 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.11 % | +0.04 pp |
| Achieved Occupancy (%) | 6.06 % | 6.06 % | +0.00 pp | 6.24 % | +0.18 pp |
| Registers/Thread | 44.00 register/thread | 44.00 register/thread | +0 | 214.00 register/thread | +170 |
| Waves/SM | 0.07 | 0.07 | +0.00 | 0.36 | +0.29 |
| No Eligible (%) | 88.70 % | 87.64 % | -1.06 pp | 74.56 % | -14.14 pp |
| IPC Active | 0.47 inst/cycle | 0.46 inst/cycle | -0.01 | 1.10 inst/cycle | +0.63 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_and_fix_block_k_for_tiny_k_gemm | 1.002× | 1.047× | ✓ |  |
| B | persistent_kernel_one_block_per_sm | 1.038× | 1.065× | ✓ | **✓ best** |
| C | reduce_tile_size_and_fix_block_k_to_match_k_dimension | 1.029× | 1.026× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | persistent_kernel... | Δ | reduce_tile_size_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 3420 ns | -160 ns (-4.5%) | 3360 ns | -220 ns (-6.1%) | 3490 ns | -90 ns (-2.5%) |
| SM Throughput (%) | 4.14 % | 3.87 % | -0.27 pp | 3.51 % | -0.63 pp | 4.07 % | -0.07 pp |
| Memory Throughput (%) | 17.59 % | 16.45 % | -1.14 pp | 16.90 % | -0.69 pp | 17.30 % | -0.29 pp |
| DRAM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp | 0.08 % | +0.01 pp | 0.08 % | +0.01 pp |
| Achieved Occupancy (%) | 6.06 % | 6.06 % | +0.00 pp | 6.18 % | +0.12 pp | 6.07 % | +0.01 pp |
| Registers/Thread | 44.00 register/thread | 44.00 register/thread | +0 | 48.00 register/thread | +4 | 44.00 register/thread | +0 |
| Waves/SM | 0.07 | 0.07 | +0.00 | 0.07 | +0.00 | 0.07 | +0.00 |
| No Eligible (%) | 88.70 % | 87.91 % | -0.79 pp | 89.88 % | +1.18 pp | 88.07 % | -0.63 pp |
| IPC Active | 0.47 inst/cycle | 0.46 inst/cycle | -0.01 | 0.40 inst/cycle | -0.07 | 0.44 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0655× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_outer_product_accumulation | 1.146× | 1.432× | ✓ |  |
| B | fix_shared_memory_bank_conflicts_via_smem_padding_and_block_k_equals_k | 1.151× | 1.432× | ✓ | **✓ best** |
| C | increase_block_n_to_reduce_shared_memory_bank_conflicts | 1.028× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | fix_shared_memory... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2500 ns | -1080 ns (-30.2%) | 2500 ns | -1080 ns (-30.2%) | 3620 ns | +40 ns (+1.1%) |
| SM Throughput (%) | 4.14 % | 2.20 % | -1.94 pp | 2.28 % | -1.86 pp | 4.16 % | +0.02 pp |
| Memory Throughput (%) | 17.59 % | 3.98 % | -13.61 pp | 3.86 % | -13.73 pp | 17.64 % | +0.05 pp |
| DRAM Throughput (%) | 0.07 % | 0.09 % | +0.02 pp | 0.08 % | +0.01 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.06 % | 6.16 % | +0.10 pp | 6.17 % | +0.11 pp | 6.08 % | +0.02 pp |
| Registers/Thread | 44.00 register/thread | 30.00 register/thread | -14 | 30.00 register/thread | -14 | 44.00 register/thread | +0 |
| Waves/SM | 0.07 | 0.05 | -0.02 | 0.05 | -0.02 | 0.07 | +0.00 |
| No Eligible (%) | 88.70 % | 90.67 % | +1.97 pp | 89.15 % | +0.45 pp | 87.46 % | -1.24 pp |
| IPC Active | 0.47 inst/cycle | 0.43 inst/cycle | -0.04 | 0.32 inst/cycle | -0.15 | 0.47 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.432× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_uncoalesced_global_access_via_vectorized_loads_and_wider_block_n | 1.156× | 1.288× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.004× | 1.065× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_g... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2780 ns | -800 ns (-22.3%) | 3360 ns | -220 ns (-6.1%) |
| SM Throughput (%) | 4.14 % | 3.03 % | -1.11 pp | 4.30 % | +0.16 pp |
| Memory Throughput (%) | 17.59 % | 3.47 % | -14.12 pp | 18.24 % | +0.65 pp |
| DRAM Throughput (%) | 0.07 % | 0.08 % | +0.01 pp | 0.08 % | +0.01 pp |
| Achieved Occupancy (%) | 6.06 % | 6.14 % | +0.08 pp | 6.08 % | +0.02 pp |
| Registers/Thread | 44.00 register/thread | 32.00 register/thread | -12 | 44.00 register/thread | +0 |
| Waves/SM | 0.07 | 0.05 | -0.02 | 0.07 | +0.00 |
| No Eligible (%) | 88.70 % | 85.83 % | -2.87 pp | 88.70 % | +0.00 pp |
| IPC Active | 0.47 inst/cycle | 0.56 inst/cycle | +0.09 | 0.48 inst/cycle | +0.01 |

---

## Iteration 5

**Best so far**: 1.432× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesced_interleaved_load_for_stride2_a_and_b_matrices | 1.138× | 1.415× | ✓ |  |
| D | outer_product_accumulation_with_hardcoded_static_range_and_wider_block_n | 1.176× | 1.288× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesced_interle... | Δ | outer_product_acc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2530 ns | -1050 ns (-29.3%) | 2780 ns | -800 ns (-22.3%) |
| SM Throughput (%) | 4.14 % | 3.07 % | -1.07 pp | 1.59 % | -2.55 pp |
| Memory Throughput (%) | 17.59 % | 3.56 % | -14.03 pp | 3.22 % | -14.37 pp |
| DRAM Throughput (%) | 0.07 % | 0.09 % | +0.02 pp | 0.11 % | +0.04 pp |
| Achieved Occupancy (%) | 6.06 % | 6.16 % | +0.10 pp | 1.56 % | -4.50 pp |
| Registers/Thread | 44.00 register/thread | 31.00 register/thread | -13 | 64.00 register/thread | +20 |
| Waves/SM | 0.07 | 0.05 | -0.02 | 0.02 | -0.05 |
| No Eligible (%) | 88.70 % | 85.80 % | -2.90 pp | 75.43 % | -13.27 pp |
| IPC Active | 0.47 inst/cycle | 0.54 inst/cycle | +0.07 | 0.20 inst/cycle | -0.27 |

---

## Cumulative Progress

```
Baseline:  3580 ns  (1.000×)
Iter 2     3360 ns  (1.065×)  — persistent_kernel_one_block_per_sm
Iter 3     2500 ns  (1.432×)  — fix_shared_memory_bank_conflicts_via_smem_padding_and_block_k_equals_k

Final best: 2780 ns  (1.2878×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,16 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=1, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=1, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_t_view_7(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 16
+    BLOCK_N : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
     A = arg_A
     B = arg_B
@@ -54,30 +63,18 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 2*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 512*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 2*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+    # CRITICAL: use literal 2, NOT tl.static_range(K)
+    # K is an int32 tensor at runtime -> AssertionError in tl.static_range
+    for k in tl.static_range(2):
+        # A strides: stride_am=2, stride_ak=1 -> load address is offs_a_m * 2 + k
+        a_col = tl.load(A + offs_a_m * 2 + k)      # shape [BLOCK_M]
+        # B strides: stride_bk=1, stride_bn=2 -> load address is k * 1 + offs_b_n * 2
+        b_row = tl.load(B + k * 1 + offs_b_n * 2)  # shape [BLOCK_N]
+        # Register-only outer product, no shared memory
+        acc = acc + a_col[:, None].to(ACC_TYPE) * b_row[None, :].to(ACC_TYPE)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +86,6 @@

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
