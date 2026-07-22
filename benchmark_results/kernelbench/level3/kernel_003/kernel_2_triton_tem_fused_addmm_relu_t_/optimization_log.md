# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4030000 ns — **Best latency**: 3950000 ns (**1.0203×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`smem_bank_conflict_relief_via_autotune_with_correct_dynamic_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_2_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.65
- **Registers/thread**: 168
- **Duration**: 4030000 ns
- **SM Throughput**: 37.45 % | **Memory**: 84.49 % | **DRAM**: 5.52 %
- **Achieved Occupancy**: 12.26 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.014× | 1.015× | ✓ |  |
| B | tma_loads_via_autotune_async_pipeline | 1.015× | 0.694× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_sizes | 0.971× | 0.988× | ✓ |  |
| D | autotune_tile_sizes_and_pipeline_stages | 0.504× | 0.504× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_via_aut... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4030000 ns | 3970000 ns | -60000 ns (-1.5%) | 5810000 ns | +1780000 ns (+44.2%) | 4080000 ns | +50000 ns (+1.2%) | 7990000 ns | +3960000 ns (+98.3%) |
| SM Throughput (%) | 37.45 % | 37.91 % | +0.46 pp | 25.90 % | -11.55 pp | 37.04 % | -0.41 pp | 37.38 % | -0.07 pp |
| Memory Throughput (%) | 84.49 % | 86.10 % | +1.61 pp | 58.80 % | -25.69 pp | 84.52 % | +0.03 pp | 85.24 % | +0.75 pp |
| DRAM Throughput (%) | 5.52 % | 5.62 % | +0.10 pp | 3.87 % | -1.65 pp | 5.79 % | +0.27 pp | 5.46 % | -0.06 pp |
| Achieved Occupancy (%) | 12.26 % | 12.25 % | -0.01 pp | 6.25 % | -6.01 pp | 12.25 % | -0.01 pp | 12.50 % | +0.24 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 255.00 register/thread | +87 | 255.00 register/thread | +87 | 254.00 register/thread | +86 |
| Waves/SM | 0.65 | 0.97 | +0.32 | 1.94 | +1.29 | 0.97 | +0.32 | 1.94 | +1.29 |
| No Eligible (%) | 61.23 % | 61.02 % | -0.21 pp | 70.46 % | +9.23 pp | 61.91 % | +0.68 pp | 61.30 % | +0.07 pp |
| IPC Active | 1.55 inst/cycle | 1.56 inst/cycle | +0.01 | 1.18 inst/cycle | -0.37 | 1.52 inst/cycle | -0.03 | 1.55 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_block_m_with_cotune | 0.989× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4030000 ns | 4060000 ns | +30000 ns (+0.7%) |
| SM Throughput (%) | 37.45 % | 37.25 % | -0.20 pp |
| Memory Throughput (%) | 84.49 % | 83.96 % | -0.53 pp |
| DRAM Throughput (%) | 5.52 % | 5.76 % | +0.24 pp |
| Achieved Occupancy (%) | 12.26 % | 12.27 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.65 | 0.65 | +0.00 |
| No Eligible (%) | 61.23 % | 61.65 % | +0.42 pp |
| IPC Active | 1.55 inst/cycle | 1.53 inst/cycle | -0.02 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_correctness_then_autotune_tile_sizes_with_dynamic_grid | 0.986× | 0.993× | ✓ |  |
| B | tma_loads_via_tensor_descriptor_async_pipeline | 1.020× | 1.020× | ✓ | **✓ best** |
| C | memory_coalescing_b_tile_transposed_load_with_maxnreg_cap | 0.990× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_correctness_t... | Δ | tma_loads_via_ten... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4030000 ns | 4060000 ns | +30000 ns (+0.7%) | 3950000 ns | -80000 ns (-2.0%) | 4060000 ns | +30000 ns (+0.7%) |
| SM Throughput (%) | 37.45 % | 37.21 % | -0.24 pp | 38.42 % | +0.97 pp | 37.20 % | -0.25 pp |
| Memory Throughput (%) | 84.49 % | 83.89 % | -0.60 pp | 86.37 % | +1.88 pp | 83.87 % | -0.62 pp |
| DRAM Throughput (%) | 5.52 % | 5.77 % | +0.25 pp | 5.93 % | +0.41 pp | 5.76 % | +0.24 pp |
| Achieved Occupancy (%) | 12.26 % | 12.27 % | +0.01 pp | 12.26 % | +0.00 pp | 12.27 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.65 | 0.65 | +0.00 | 0.65 | +0.00 | 0.65 | +0.00 |
| No Eligible (%) | 61.23 % | 61.65 % | +0.42 pp | 60.56 % | -0.67 pp | 61.65 % | +0.42 pp |
| IPC Active | 1.55 inst/cycle | 1.53 inst/cycle | -0.02 | 1.58 inst/cycle | +0.03 | 1.53 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.0203× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_to_eliminate_smem_bank_conflicts | 1.018× | 1.020× | ✓ |  |
| C | fix_b_tile_load_correctness_and_eliminate_smem_bank_conflicts_via_standard_kn_load | 0.991× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | fix_b_tile_load_c... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4030000 ns | 3950000 ns | -80000 ns (-2.0%) | 4060000 ns | +30000 ns (+0.7%) |
| SM Throughput (%) | 37.45 % | 38.42 % | +0.97 pp | 37.23 % | -0.22 pp |
| Memory Throughput (%) | 84.49 % | 86.38 % | +1.89 pp | 83.92 % | -0.57 pp |
| DRAM Throughput (%) | 5.52 % | 5.92 % | +0.40 pp | 5.76 % | +0.24 pp |
| Achieved Occupancy (%) | 12.26 % | 12.26 % | +0.00 pp | 12.27 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.65 | 0.65 | +0.00 | 0.65 | +0.00 |
| No Eligible (%) | 61.23 % | 60.56 % | -0.67 pp | 61.65 % | +0.42 pp |
| IPC Active | 1.55 inst/cycle | 1.58 inst/cycle | +0.03 | 1.53 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0203× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | smem_bank_conflict_relief_via_autotune_with_correct_dynamic_grid | 1.017× | 1.020× | ✓ |  |
| C | autotune_tile_sizes_with_dynamic_grid_and_smem_padding_for_bank_conflict_relief | 0.991× | 0.993× | ✓ |  |
| D | tma_loads_via_make_tensor_descriptor | 1.020× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | smem_bank_conflic... | Δ | autotune_tile_siz... | Δ | tma_loads_via_mak... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4030000 ns | 3950000 ns | -80000 ns (-2.0%) | 4060000 ns | +30000 ns (+0.7%) | 3950000 ns | -80000 ns (-2.0%) |
| SM Throughput (%) | 37.45 % | 38.41 % | +0.96 pp | 37.19 % | -0.26 pp | 38.40 % | +0.95 pp |
| Memory Throughput (%) | 84.49 % | 86.37 % | +1.88 pp | 83.84 % | -0.65 pp | 86.34 % | +1.85 pp |
| DRAM Throughput (%) | 5.52 % | 5.92 % | +0.40 pp | 5.76 % | +0.24 pp | 5.93 % | +0.41 pp |
| Achieved Occupancy (%) | 12.26 % | 12.26 % | +0.00 pp | 12.27 % | +0.01 pp | 12.26 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.65 | 0.65 | +0.00 | 0.65 | +0.00 | 0.65 | +0.00 |
| No Eligible (%) | 61.23 % | 60.56 % | -0.67 pp | 61.65 % | +0.42 pp | 60.57 % | -0.66 pp |
| IPC Active | 1.55 inst/cycle | 1.58 inst/cycle | +0.03 | 1.53 inst/cycle | -0.02 | 1.58 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  4030000 ns  (1.000×)
Iter 3     3950000 ns  (1.020×)  — tma_loads_via_tensor_descriptor_async_pipeline

Final best: 3950000 ns  (1.0203×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_addmm_relu_t_2(arg_A, arg_B, in_ptr2, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 128
-    BLOCK_N : tl.constexpr = 64
+    BLOCK_M : tl.constexpr = 64
+    BLOCK_N : tl.constexpr = 128
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -54,24 +55,32 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointers once before the loop for TMA-compatible loads
+    A_block_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)  # row-major: K is innermost (stride_ak=1)
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)  # CORRECT: dim-0 (K, stride=1) is fastest in memory
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 16384*idx_m
-        a = tl.load(A + (xindex))
-
-        # Load B tile: shape [BLOCK_K, BLOCK_N]
-        # B[k,n] = B_ptr + k*stride_bk + n*stride_bn = B_ptr + k*1 + n*16384
-        b = tl.load(B + (b_k_idx_vals + 16384 * offs_b_n[None, :]))
-
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -87,7 +96,6 @@

     tmp2 = tl.full([1], 0, tl.int32)
     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 16384*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
