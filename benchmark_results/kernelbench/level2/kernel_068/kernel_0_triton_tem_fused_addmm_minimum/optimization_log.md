# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_minimum_sub_t_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4020000 ns — **Best latency**: 3970000 ns (**1.0126×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`tma_loads_via_block_ptr_for_bank_conflict_elimination`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_minimum_sub_t_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_068/kernel_0_triton_tem_fused_addmm_minimum/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.65
- **Registers/thread**: 168
- **Duration**: 4020000 ns
- **SM Throughput**: 37.52 % | **Memory**: 84.65 % | **DRAM**: 5.53 %
- **Achieved Occupancy**: 12.26 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 0.998× | ✓ |  |
| B | increase_block_k_reduce_block_m_add_autotune_to_eliminate_bank_conflicts | 0.984× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | increase_block_k_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4020000 ns | 4030000 ns | +10000 ns (+0.2%) | 4080000 ns | +60000 ns (+1.5%) |
| SM Throughput (%) | 37.52 % | 37.53 % | +0.01 pp | 37.08 % | -0.44 pp |
| Memory Throughput (%) | 84.65 % | 84.68 % | +0.03 pp | 84.62 % | -0.03 pp |
| DRAM Throughput (%) | 5.53 % | 5.52 % | -0.01 pp | 5.79 % | +0.26 pp |
| Achieved Occupancy (%) | 12.26 % | 12.26 % | +0.00 pp | 12.26 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.65 | 0.65 | +0.00 | 0.97 | +0.32 |
| No Eligible (%) | 61.22 % | 61.22 % | +0.00 pp | 61.90 % | +0.68 pp |
| IPC Active | 1.55 inst/cycle | 1.55 inst/cycle | +0.00 | 1.52 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_matrix_b_load_index_to_eliminate_shared_memory_bank_conflicts | 1.004× | 1.002× | ✓ |  |
| B | tma_loads_via_block_ptr_for_bank_conflict_elimination | 1.013× | 1.013× | ✓ | **✓ best** |
| C | fix_matrix_b_load_correct_row_major_indexing_to_eliminate_bank_conflicts | 0.985× | 0.985× | ✓ |  |
| E | add_triton_autotune | 1.000× | 0.920× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_matrix_b_load... | Δ | tma_loads_via_blo... | Δ | fix_matrix_b_load... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4020000 ns | 4010000 ns | -10000 ns (-0.2%) | 3970000 ns | -50000 ns (-1.2%) | 4080000 ns | +60000 ns (+1.5%) | 4370000 ns | +350000 ns (+8.7%) |
| SM Throughput (%) | 37.52 % | 37.70 % | +0.18 pp | 38.16 % | +0.64 pp | 37.06 % | -0.46 pp | 34.30 % | -3.22 pp |
| Memory Throughput (%) | 84.65 % | 85.06 % | +0.41 pp | 85.89 % | +1.24 pp | 84.57 % | -0.08 pp | 78.01 % | -6.64 pp |
| DRAM Throughput (%) | 5.53 % | 5.55 % | +0.02 pp | 5.60 % | +0.07 pp | 5.79 % | +0.26 pp | 5.09 % | -0.44 pp |
| Achieved Occupancy (%) | 12.26 % | 12.26 % | +0.00 pp | 12.26 % | +0.00 pp | 12.26 % | +0.00 pp | 12.26 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 | 162.00 register/thread | -6 |
| Waves/SM | 0.65 | 0.65 | +0.00 | 0.65 | +0.00 | 0.97 | +0.32 | 0.65 | +0.00 |
| No Eligible (%) | 61.22 % | 61.17 % | -0.05 pp | 60.74 % | -0.48 pp | 61.90 % | +0.68 pp | 64.58 % | +3.36 pp |
| IPC Active | 1.55 inst/cycle | 1.55 inst/cycle | +0.00 | 1.57 inst/cycle | +0.02 | 1.52 inst/cycle | -0.03 | 1.42 inst/cycle | -0.13 |

---

## Iteration 3

**Best so far**: 1.0126× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_for_conflict_free_shared_memory_layout | 1.002× | 0.922× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4020000 ns | 4360000 ns | +340000 ns (+8.5%) |
| SM Throughput (%) | 37.52 % | 34.35 % | -3.17 pp |
| Memory Throughput (%) | 84.65 % | 78.18 % | -6.47 pp |
| DRAM Throughput (%) | 5.53 % | 5.10 % | -0.43 pp |
| Achieved Occupancy (%) | 12.26 % | 12.26 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 157.00 register/thread | -11 |
| Waves/SM | 0.65 | 0.65 | +0.00 |
| No Eligible (%) | 61.22 % | 64.66 % | +3.44 pp |
| IPC Active | 1.55 inst/cycle | 1.41 inst/cycle | -0.14 |

---

## Iteration 4

**Best so far**: 1.0126× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | convert_matrix_a_scalar_load_to_block_ptr_for_coalesced_tma_access | 1.011× | 0.944× | ✓ |  |
| C | fix_block_ptr_order_for_conflict_free_smem_layout | 1.002× | 0.922× | ✓ |  |


### Metric Comparison

| Metric | Baseline | convert_matrix_a_... | Δ | fix_block_ptr_ord... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4020000 ns | 4260000 ns | +240000 ns (+6.0%) | 4360000 ns | +340000 ns (+8.5%) |
| SM Throughput (%) | 37.52 % | 35.13 % | -2.39 pp | 34.34 % | -3.18 pp |
| Memory Throughput (%) | 84.65 % | 80.07 % | -4.58 pp | 78.15 % | -6.50 pp |
| DRAM Throughput (%) | 5.53 % | 5.23 % | -0.30 pp | 5.10 % | -0.43 pp |
| Achieved Occupancy (%) | 12.26 % | 12.25 % | -0.01 pp | 12.26 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 157.00 register/thread | -11 |
| Waves/SM | 0.65 | 0.65 | +0.00 | 0.65 | +0.00 |
| No Eligible (%) | 61.22 % | 63.89 % | +2.67 pp | 64.66 % | +3.44 pp |
| IPC Active | 1.55 inst/cycle | 1.44 inst/cycle | -0.11 | 1.41 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  4020000 ns  (1.000×)
Iter 2     3970000 ns  (1.013×)  — tma_loads_via_block_ptr_for_bank_conflict_elimination

Final best: 3970000 ns  (1.0126×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_minimum_sub_t_0(arg_A, arg_B, in_ptr2, in_ptr3, out_ptr1):
     EVEN_K : tl.constexpr = True
@@ -57,21 +67,26 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(1, 16384),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 16384*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 16384*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        b = tl.load(B_block_ptr)
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -92,8 +107,6 @@

     tmp4 = tl.minimum(tmp1, tmp3)
     tmp5 = tmp4 - tmp3
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 16384*idx_m, [BLOCK_M, BLOCK_N])), tmp5, mask)
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
