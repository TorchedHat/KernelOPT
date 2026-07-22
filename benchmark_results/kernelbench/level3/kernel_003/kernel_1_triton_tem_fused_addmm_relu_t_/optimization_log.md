# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 176900 ns — **Best latency**: 200290 ns (**0.8832×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_1_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 176900 ns
- **SM Throughput**: 27.02 % | **Memory**: 60.88 % | **DRAM**: 0.97 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 69.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | autotune_tile_sizes_to_fix_bank_conflicts_and_register_pressure | 1.178× | 0.889× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 176900 ns | 198880 ns | +21980 ns (+12.4%) |
| SM Throughput (%) | 27.02 % | 24.04 % | -2.98 pp |
| Memory Throughput (%) | 60.88 % | 54.61 % | -6.27 pp |
| DRAM Throughput (%) | 0.97 % | 0.86 % | -0.11 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 |
| No Eligible (%) | 69.53 % | 74.72 % | +5.19 pp |
| IPC Active | 1.22 inst/cycle | 1.01 inst/cycle | -0.21 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_coalesced_load_eliminate_bank_conflicts | 1.208× | 0.883× | ✓ | **✓ best** |
| C | fix_b_matrix_strided_load_to_eliminate_bank_conflicts | 1.174× | 0.896× | ✓ |  |
| D | increase_block_k_and_fix_b_load_indexing_to_reduce_register_pressure | 1.006× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_coal... | Δ | fix_b_matrix_stri... | Δ | increase_block_k_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 176900 ns | 200290 ns | +23390 ns (+13.2%) | 197500 ns | +20600 ns (+11.6%) | 183580 ns | +6680 ns (+3.8%) |
| SM Throughput (%) | 27.02 % | 23.81 % | -3.21 pp | 24.01 % | -3.01 pp | 25.79 % | -1.23 pp |
| Memory Throughput (%) | 60.88 % | 54.09 % | -6.79 pp | 54.54 % | -6.34 pp | 58.53 % | -2.35 pp |
| DRAM Throughput (%) | 0.97 % | 0.86 % | -0.11 pp | 0.87 % | -0.10 pp | 0.94 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.32 | +0.00 | 0.32 | +0.00 |
| No Eligible (%) | 69.53 % | 75.00 % | +5.47 pp | 74.73 % | +5.20 pp | 72.16 % | +2.63 pp |
| IPC Active | 1.22 inst/cycle | 1.00 inst/cycle | -0.22 | 1.01 inst/cycle | -0.21 | 1.11 inst/cycle | -0.11 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_via_transpose_to_eliminate_bank_conflicts | 1.002× | 0.963× | ✓ |  |
| C | replace_manual_ptrs_with_block_ptr_api_to_eliminate_bank_conflicts | 1.131× | 0.852× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | replace_manual_pt... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 176900 ns | 183620 ns | +6720 ns (+3.8%) | 207740 ns | +30840 ns (+17.4%) |
| SM Throughput (%) | 27.02 % | 25.83 % | -1.19 pp | 23.18 % | -3.84 pp |
| Memory Throughput (%) | 60.88 % | 58.62 % | -2.26 pp | 52.12 % | -8.76 pp |
| DRAM Throughput (%) | 0.97 % | 0.94 % | -0.03 pp | 0.83 % | -0.14 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.32 | +0.00 |
| No Eligible (%) | 69.53 % | 72.21 % | +2.68 pp | 75.74 % | +6.21 pp |
| IPC Active | 1.22 inst/cycle | 1.11 inst/cycle | -0.11 | 0.97 inst/cycle | -0.25 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_with_correct_order_to_eliminate_bank_conflicts | 0.993× | 0.978× | ✓ |  |
| D | tma_loads_via_tensor_descriptor_hopper | 1.134× | 1.139× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | tma_loads_via_ten... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 176900 ns | 180960 ns | +4060 ns (+2.3%) | 155330 ns | -21570 ns (-12.2%) |
| SM Throughput (%) | 27.02 % | 25.99 % | -1.03 pp | 31.18 % | +4.16 pp |
| Memory Throughput (%) | 60.88 % | 59.22 % | -1.66 pp | 69.21 % | +8.33 pp |
| DRAM Throughput (%) | 0.97 % | 0.95 % | -0.02 pp | 1.10 % | +0.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 232.00 register/thread | +64 |
| Waves/SM | 0.32 | 0.32 | +0.00 | 0.48 | +0.16 |
| No Eligible (%) | 69.53 % | 71.61 % | +2.08 pp | 67.24 % | -2.29 pp |
| IPC Active | 1.22 inst/cycle | 1.14 inst/cycle | -0.08 | 1.31 inst/cycle | +0.09 |

---

## Cumulative Progress

```
Baseline:  176900 ns  (1.000×)

Final best: 200290 ns  (0.8832×)
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
 def triton_tem_fused_addmm_relu_t_1(arg_A, arg_B, in_ptr2, out_ptr1):
     EVEN_K : tl.constexpr = True
@@ -54,27 +64,19 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 1024*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1024*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        k_start = k_idx * BLOCK_K
+        offs_k_cur = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
+        # A tile: [BLOCK_M, BLOCK_K], stride_am=1024, stride_ak=1
+        a_ptrs = A + offs_a_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak
+        a = tl.load(a_ptrs)  # shape [BLOCK_M, BLOCK_K]
+        # B tile: load transposed [BLOCK_N, BLOCK_K] for coalesced access, then transpose back
+        b_ptrs = B + offs_b_n[:, None] * stride_bn + offs_k_cur[None, :] * stride_bk
+        b_T = tl.load(b_ptrs)  # shape [BLOCK_N, BLOCK_K] — coalesced in K dim
+        b = tl.trans(b_T)       # shape [BLOCK_K, BLOCK_N] — correct for tl.dot(a, b)
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -90,8 +92,6 @@

     tmp2 = tl.full([1], 0, tl.int32)
     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1024*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
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
