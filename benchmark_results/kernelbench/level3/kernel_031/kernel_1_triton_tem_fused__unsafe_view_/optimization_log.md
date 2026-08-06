# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 200900 ns — **Best latency**: 201760 ns (**0.9957×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_1_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (768,1,1), Waves/SM: 2.91
- **Registers/thread**: 255
- **Duration**: 200900 ns
- **SM Throughput**: 37.07 % | **Memory**: 56.35 % | **DRAM**: 5.45 %
- **Achieved Occupancy**: 12.24 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 200900 ns | 200740 ns | -160 ns (-0.1%) |
| SM Throughput (%) | 37.07 % | 37.10 % | +0.03 pp |
| Memory Throughput (%) | 56.35 % | 56.40 % | +0.05 pp |
| DRAM Throughput (%) | 5.45 % | 5.44 % | -0.01 pp |
| Achieved Occupancy (%) | 12.24 % | 12.23 % | -0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 2.91 | 2.91 | +0.00 |
| No Eligible (%) | 61.12 % | 61.22 % | +0.10 pp |
| IPC Active | 1.55 inst/cycle | 1.56 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_load_index_and_reduce_register_pressure_via_autotune | 1.002× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 200900 ns | 201060 ns | +160 ns (+0.1%) |
| SM Throughput (%) | 37.07 % | 37.11 % | +0.04 pp |
| Memory Throughput (%) | 56.35 % | 56.41 % | +0.06 pp |
| DRAM Throughput (%) | 5.45 % | 5.44 % | -0.01 pp |
| Achieved Occupancy (%) | 12.24 % | 12.25 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 2.91 | 2.91 | +0.00 |
| No Eligible (%) | 61.12 % | 61.26 % | +0.14 pp |
| IPC Active | 1.55 inst/cycle | 1.55 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | autotune_tile_sizes_with_dynamic_grid_and_eviction_policies | 1.032× | 0.996× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 200900 ns | 201760 ns | +860 ns (+0.4%) |
| SM Throughput (%) | 37.07 % | 37.05 % | -0.02 pp |
| Memory Throughput (%) | 56.35 % | 56.31 % | -0.04 pp |
| DRAM Throughput (%) | 5.45 % | 5.41 % | -0.04 pp |
| Achieved Occupancy (%) | 12.24 % | 12.24 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 2.91 | 2.91 | +0.00 |
| No Eligible (%) | 61.12 % | 61.26 % | +0.14 pp |
| IPC Active | 1.55 inst/cycle | 1.56 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_make_block_ptr | 1.017× | 1.009× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_mak... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 200900 ns | 199100 ns | -1800 ns (-0.9%) |
| SM Throughput (%) | 37.07 % | 37.23 % | +0.16 pp |
| Memory Throughput (%) | 56.35 % | 56.30 % | -0.05 pp |
| DRAM Throughput (%) | 5.45 % | 4.88 % | -0.57 pp |
| Achieved Occupancy (%) | 12.24 % | 12.36 % | +0.12 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 2.91 | 2.91 | +0.00 |
| No Eligible (%) | 61.12 % | 60.95 % | -0.17 pp |
| IPC Active | 1.55 inst/cycle | 1.57 inst/cycle | +0.02 |

---

## Iteration 5

**Best so far**: 1.0008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  200900 ns  (1.000×)
Iter 1     200740 ns  (1.001×)  — tf32_tensor_cores

Final best: 201760 ns  (0.9957×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
@@ -58,23 +72,23 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 128*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
-        xindex = idx_n + 384*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # B stored as (384,128) strides (128,1): B[n,k] = n*128 + k*1
+        # Tile element [k,n] -> offset = k*stride_bk + n*stride_bn = k*1 + n*128
+        xindex = idx_m * 1 + idx_n * 128
+        b = tl.load(B + xindex, eviction_policy='evict_first')
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +100,6 @@

     # inductor generates a suffix
     xindex = idx_n + 384*idx_m
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
