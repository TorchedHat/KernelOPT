# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 962020 ns — **Best latency**: 959070 ns (**1.0031×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`tma_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_3_triton_tem_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 5.94
- **Registers/thread**: 96
- **Duration**: 962020 ns
- **SM Throughput**: 20.27 % | **Memory**: 96.89 % | **DRAM**: 21.05 %
- **Achieved Occupancy**: 30.01 %
- **IPC Active**: 0.82 inst/cycle | **No Eligible**: 79.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |
| B | tma_loads_with_block_ptr_to_eliminate_bank_conflicts | 1.001× | 1.003× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_improve_occupancy | 0.892× | 0.598× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_bl... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 962020 ns | 962140 ns | +120 ns (+0.0%) | 959300 ns | -2720 ns (-0.3%) | 1610000 ns | +647980 ns (+67.4%) |
| SM Throughput (%) | 20.27 % | 20.30 % | +0.03 pp | 20.41 % | +0.14 pp | 22.23 % | +1.96 pp |
| Memory Throughput (%) | 96.89 % | 97.00 % | +0.11 pp | 97.26 % | +0.37 pp | 95.70 % | -1.19 pp |
| DRAM Throughput (%) | 21.05 % | 21.04 % | -0.01 pp | 21.09 % | +0.04 pp | 12.55 % | -8.50 pp |
| Achieved Occupancy (%) | 30.01 % | 30.01 % | +0.00 pp | 35.85 % | +5.84 pp | 18.45 % | -11.56 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 80.00 register/thread | -16 | 160.00 register/thread | +64 |
| Waves/SM | 5.94 | 5.94 | +0.00 | 4.95 | -0.99 | 9.90 | +3.96 |
| No Eligible (%) | 79.43 % | 79.43 % | +0.00 pp | 79.32 % | -0.11 pp | 77.39 % | -2.04 pp |
| IPC Active | 0.82 inst/cycle | 0.82 inst/cycle | +0.00 | 0.83 inst/cycle | +0.01 | 0.90 inst/cycle | +0.08 |

---

## Iteration 2

**Best so far**: 1.0028× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0028× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_break_shared_memory_bank_conflict_alignment | 0.951× | 0.953× | ✓ |  |
| B | tma_loads | 1.001× | 1.003× | ✓ | **✓ best** |
| C | memory_coalescing_transpose_b_tile_for_contiguous_shared_memory_layout | 0.998× | 1.000× | ✓ |  |
| D | fix_b_load_stride_via_transpose_to_eliminate_12way_bank_conflicts | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | tma_loads | Δ | memory_coalescing... | Δ | fix_b_load_stride... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 962020 ns | 1010000 ns | +47980 ns (+5.0%) | 959070 ns | -2950 ns (-0.3%) | 961820 ns | -200 ns (-0.0%) | 962240 ns | +220 ns (+0.0%) |
| SM Throughput (%) | 20.27 % | 20.49 % | +0.22 pp | 20.40 % | +0.13 pp | 20.29 % | +0.02 pp | 20.29 % | +0.02 pp |
| Memory Throughput (%) | 96.89 % | 95.17 % | -1.72 pp | 97.22 % | +0.33 pp | 96.96 % | +0.07 pp | 96.98 % | +0.09 pp |
| DRAM Throughput (%) | 21.05 % | 20.18 % | -0.87 pp | 21.10 % | +0.05 pp | 21.05 % | +0.00 pp | 21.04 % | -0.01 pp |
| Achieved Occupancy (%) | 30.01 % | 30.13 % | +0.12 pp | 35.85 % | +5.84 pp | 30.01 % | +0.00 pp | 30.01 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 | 80.00 register/thread | -16 | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 5.94 | 5.94 | +0.00 | 4.95 | -0.99 | 5.94 | +0.00 | 5.94 | +0.00 |
| No Eligible (%) | 79.43 % | 79.22 % | -0.21 pp | 79.32 % | -0.11 pp | 79.43 % | +0.00 pp | 79.44 % | +0.01 pp |
| IPC Active | 0.82 inst/cycle | 0.83 inst/cycle | +0.01 | 0.83 inst/cycle | +0.01 | 0.82 inst/cycle | +0.00 | 0.82 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0031× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_smem_layout_order_to_eliminate_12way_bank_conflicts | 1.000× | 1.003× | ✓ |  |
| B | reduce_block_k_to_16_for_square_b_tile_eliminating_bank_conflicts | 0.951× | 0.953× | ✓ |  |
| C | cache_b_in_shared_memory_with_padding_to_eliminate_bank_conflicts | 1.000× | 0.999× | ✓ |  |
| D | cap_registers_with_maxnreg_to_eliminate_local_memory_spill | 0.950× | 0.962× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_smem_layout... | Δ | reduce_block_k_to... | Δ | cache_b_in_shared... | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 962020 ns | 959170 ns | -2850 ns (-0.3%) | 1010000 ns | +47980 ns (+5.0%) | 962590 ns | +570 ns (+0.1%) | 1000000 ns | +37980 ns (+3.9%) |
| SM Throughput (%) | 20.27 % | 20.40 % | +0.13 pp | 20.49 % | +0.22 pp | 20.29 % | +0.02 pp | 19.74 % | -0.53 pp |
| Memory Throughput (%) | 96.89 % | 97.23 % | +0.34 pp | 95.16 % | -1.73 pp | 96.99 % | +0.10 pp | 95.63 % | -1.26 pp |
| DRAM Throughput (%) | 21.05 % | 21.10 % | +0.05 pp | 20.19 % | -0.86 pp | 21.03 % | -0.02 pp | 20.57 % | -0.48 pp |
| Achieved Occupancy (%) | 30.01 % | 35.85 % | +5.84 pp | 30.13 % | +0.12 pp | 30.01 % | +0.00 pp | 46.66 % | +16.65 pp |
| Registers/Thread | 96.00 register/thread | 80.00 register/thread | -16 | 96.00 register/thread | +0 | 96.00 register/thread | +0 | 64.00 register/thread | -32 |
| Waves/SM | 5.94 | 4.95 | -0.99 | 5.94 | +0.00 | 5.94 | +0.00 | 3.71 | -2.23 |
| No Eligible (%) | 79.43 % | 79.31 % | -0.12 pp | 79.23 % | -0.20 pp | 79.43 % | +0.00 pp | 79.98 % | +0.55 pp |
| IPC Active | 0.82 inst/cycle | 0.83 inst/cycle | +0.01 | 0.83 inst/cycle | +0.01 | 0.82 inst/cycle | +0.00 | 0.80 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0031× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | hoist_full_b_matrix_into_padded_shared_memory_once | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_full_b_matr... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 962020 ns | 961730 ns | -290 ns (-0.0%) |
| SM Throughput (%) | 20.27 % | 20.29 % | +0.02 pp |
| Memory Throughput (%) | 96.89 % | 96.99 % | +0.10 pp |
| DRAM Throughput (%) | 21.05 % | 21.06 % | +0.01 pp |
| Achieved Occupancy (%) | 30.01 % | 30.01 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 |
| Waves/SM | 5.94 | 5.94 | +0.00 |
| No Eligible (%) | 79.43 % | 79.44 % | +0.01 pp |
| IPC Active | 0.82 inst/cycle | 0.82 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  962020 ns  (1.000×)
Iter 1     959300 ns  (1.003×)  — tma_loads_with_block_ptr_to_eliminate_bank_conflicts
Iter 3     959070 ns  (1.003×)  — tma_loads

Final best: 959070 ns  (1.0031×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,11 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_convolution_3(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 16
@@ -54,27 +55,35 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointers for A [M, K] with strides [480, 1] (row-major)
+    # and B [K, N] with strides [1, 480] (K-major)
+    # Using tl.make_block_ptr to enable TMA-style loads on Hopper
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),   # column-major SMEM: N-stride = 1 float => 0 bank conflicts
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
-        xindex = idx_n + 16*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
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
@@ -88,8 +97,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
