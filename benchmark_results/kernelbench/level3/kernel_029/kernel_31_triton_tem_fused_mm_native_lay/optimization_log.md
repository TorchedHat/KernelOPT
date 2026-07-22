# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_native_layer_norm_t_view_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 132450 ns — **Best latency**: 114880 ns (**1.1529×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_load_indexing_and_eliminate_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_native_layer_norm_t_view_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_31_triton_tem_fused_mm_native_lay/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (93,1,1), Waves/SM: 0.23
- **Registers/thread**: 168
- **Duration**: 132450 ns
- **SM Throughput**: 19.70 % | **Memory**: 44.33 % | **DRAM**: 1.15 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.21 inst/cycle | **No Eligible**: 69.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.001× | 1.003× | ✓ |  |
| B | tma_loads_with_autotune | 0.978× | 0.814× | ✓ |  |
| C | eviction_policies_for_streaming_vs_reused_data | 1.000× | 1.001× | ✓ |  |
| D | add_autotune_reduce_register_pressure_increase_sm_coverage | 1.002× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ | eviction_policies... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 132450 ns | 132030 ns | -420 ns (-0.3%) | 162690 ns | +30240 ns (+22.8%) | 132260 ns | -190 ns (-0.1%) | 132290 ns | -160 ns (-0.1%) |
| SM Throughput (%) | 19.70 % | 19.69 % | -0.01 pp | 16.63 % | -3.07 pp | 19.66 % | -0.04 pp | 19.69 % | -0.01 pp |
| Memory Throughput (%) | 44.33 % | 44.31 % | -0.02 pp | 64.73 % | +20.40 pp | 44.22 % | -0.11 pp | 44.31 % | -0.02 pp |
| DRAM Throughput (%) | 1.15 % | 1.12 % | -0.03 pp | 0.90 % | -0.25 pp | 1.12 % | -0.03 pp | 1.12 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 12.50 % | +6.25 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 128.00 register/thread | -40 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.35 | +0.12 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 69.72 % | 69.73 % | +0.01 pp | 76.04 % | +6.32 pp | 69.71 % | -0.01 pp | 69.73 % | +0.01 pp |
| IPC Active | 1.21 inst/cycle | 1.21 inst/cycle | +0.00 | 0.96 inst/cycle | -0.25 | 1.21 inst/cycle | +0.00 | 1.21 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0012× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0012× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_load_indexing_and_eliminate_shared_memory_bank_conflicts | 1.160× | 1.153× | ✓ | **✓ best** |
| B | shared_memory_padding_to_eliminate_bank_conflicts | 0.983× | 0.980× | ✓ |  |
| C | fix_b_matrix_transposed_load_to_eliminate_shared_memory_bank_conflicts | 1.001× | 1.000× | ✓ |  |
| D | fix_b_matrix_load_indexing_bug_then_autotune_tile_sizes | 0.997× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_indexi... | Δ | shared_memory_pad... | Δ | fix_b_matrix_tran... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 132450 ns | 114880 ns | -17570 ns (-13.3%) | 135170 ns | +2720 ns (+2.1%) | 132380 ns | -70 ns (-0.1%) | 133310 ns | +860 ns (+0.6%) |
| SM Throughput (%) | 19.70 % | 22.85 % | +3.15 pp | 19.12 % | -0.58 pp | 19.69 % | -0.01 pp | 19.56 % | -0.14 pp |
| Memory Throughput (%) | 44.33 % | 51.14 % | +6.81 pp | 46.07 % | +1.74 pp | 44.29 % | -0.04 pp | 44.01 % | -0.32 pp |
| DRAM Throughput (%) | 1.15 % | 1.28 % | +0.13 pp | 1.09 % | -0.06 pp | 1.11 % | -0.04 pp | 1.11 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 236.00 register/thread | +68 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.35 | +0.12 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 69.72 % | 66.82 % | -2.90 pp | 71.39 % | +1.67 pp | 69.76 % | +0.04 pp | 69.87 % | +0.15 pp |
| IPC Active | 1.21 inst/cycle | 1.33 inst/cycle | +0.12 | 1.14 inst/cycle | -0.07 | 1.21 inst/cycle | +0.00 | 1.20 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.1529× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_b_load_correct_index_then_autotune_tile_shapes | 0.983× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_correc... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 132450 ns | 135070 ns | +2620 ns (+2.0%) |
| SM Throughput (%) | 19.70 % | 19.11 % | -0.59 pp |
| Memory Throughput (%) | 44.33 % | 46.07 % | +1.74 pp |
| DRAM Throughput (%) | 1.15 % | 1.09 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 69.72 % | 71.36 % | +1.64 pp |
| IPC Active | 1.21 inst/cycle | 1.14 inst/cycle | -0.07 |

---

## Iteration 5

**Best so far**: 1.1529× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  132450 ns  (1.000×)
Iter 1     132290 ns  (1.001×)  — add_autotune_reduce_register_pressure_increase_sm_coverage
Iter 3     114880 ns  (1.153×)  — fix_b_load_indexing_and_eliminate_shared_memory_bank_conflicts

Final best: 114880 ns  (1.1529×)
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
 def triton_tem_fused_mm_native_layer_norm_t_view_31(arg_A, arg_B, out_ptr0):
@@ -12,7 +13,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,23 +59,20 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        # A tile: shape [BLOCK_M, BLOCK_K]
+        k_offs_a = offs_k[None, :] + (k_idx * BLOCK_K)   # [1, BLOCK_K]
+        a = tl.load(A + (offs_a_m[:, None] * 768 + k_offs_a),
+                    eviction_policy='evict_first')          # [BLOCK_M, BLOCK_K]
 
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
-        xindex = idx_n + 384*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B tile: shape [BLOCK_K, BLOCK_N]
+        # B is (N=384, K=768) layout: B[n, k] = B_ptr + n*768 + k
+        # We need B[k_global, n] for tl.dot(A[m,k], B[k,n])
+        # Address = offs_b_n[None,:]*768 + k_global_offs[:,None]
+        k_offs_b = offs_k[:, None] + (k_idx * BLOCK_K)    # [BLOCK_K, 1]
+        b = tl.load(B + (offs_b_n[None, :] * 768 + k_offs_b),
+                    eviction_policy='evict_last')           # [BLOCK_K, BLOCK_N]
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -85,9 +83,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 384*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
