# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_23` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 30340 ns — **Best latency**: 30620 ns (**0.9909×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_23`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_23_triton_tem_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 96
- **Duration**: 30340 ns
- **SM Throughput**: 10.95 % | **Memory**: 78.25 % | **DRAM**: 5.43 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 87.26 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.136× | 1.100× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.349× | 0.991× | ✓ | **✓ best** |
| C | fix_matrix_b_load_coalescing_and_autotune_tiles | 1.008× | 1.008× | ✓ |  |
| D | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 1.145× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | fix_matrix_b_load... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30340 ns | 27580 ns | -2760 ns (-9.1%) | 30620 ns | +280 ns (+0.9%) | 30110 ns | -230 ns (-0.8%) | 29920 ns | -420 ns (-1.4%) |
| SM Throughput (%) | 10.95 % | 11.04 % | +0.09 pp | 10.64 % | -0.31 pp | 10.91 % | -0.04 pp | 10.82 % | -0.13 pp |
| Memory Throughput (%) | 78.25 % | 47.07 % | -31.18 pp | 76.13 % | -2.12 pp | 77.97 % | -0.28 pp | 77.34 % | -0.91 pp |
| DRAM Throughput (%) | 5.43 % | 5.99 % | +0.56 pp | 5.37 % | -0.06 pp | 5.47 % | +0.04 pp | 5.51 % | +0.08 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.25 % | +0.01 pp | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 168.00 register/thread | +72 | 92.00 register/thread | -4 | 96.00 register/thread | +0 | 96.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.16 | -0.07 | 0.19 | -0.04 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 87.26 % | 72.52 % | -14.74 pp | 87.62 % | +0.36 pp | 87.32 % | +0.06 pp | 87.27 % | +0.01 pp |
| IPC Active | 0.51 inst/cycle | 0.55 inst/cycle | +0.04 | 0.50 inst/cycle | -0.01 | 0.51 inst/cycle | +0.00 | 0.51 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_matrix_b_transposed_load_to_eliminate_bank_conflicts_and_uncoalesced_access | 1.146× | 1.114× | ✓ |  |
| C | fix_b_matrix_shared_memory_bank_conflicts_via_transposed_load | 1.135× | 1.110× | ✓ |  |
| D | fix_matrix_b_transposed_smem_access_with_block_ptr_and_padding | 1.297× | 1.033× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_matrix_b_tran... | Δ | fix_b_matrix_shar... | Δ | fix_matrix_b_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30340 ns | 27230 ns | -3110 ns (-10.3%) | 27330 ns | -3010 ns (-9.9%) | 29380 ns | -960 ns (-3.2%) |
| SM Throughput (%) | 10.95 % | 11.25 % | +0.30 pp | 11.24 % | +0.29 pp | 11.17 % | +0.22 pp |
| Memory Throughput (%) | 78.25 % | 47.94 % | -30.31 pp | 47.92 % | -30.33 pp | 79.27 % | +1.02 pp |
| DRAM Throughput (%) | 5.43 % | 6.07 % | +0.64 pp | 6.04 % | +0.61 pp | 5.60 % | +0.17 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 96.00 register/thread | 168.00 register/thread | +72 | 168.00 register/thread | +72 | 93.00 register/thread | -3 |
| Waves/SM | 0.23 | 0.16 | -0.07 | 0.16 | -0.07 | 0.19 | -0.04 |
| No Eligible (%) | 87.26 % | 72.99 % | -14.27 pp | 72.50 % | -14.76 pp | 87.21 % | -0.05 pp |
| IPC Active | 0.51 inst/cycle | 0.55 inst/cycle | +0.04 | 0.55 inst/cycle | +0.04 | 0.52 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eliminate_smem_bank_conflicts_via_explicit_padding_in_block_ptr_loads | 1.321× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_smem_ba... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 30340 ns | 29980 ns | -360 ns (-1.2%) |
| SM Throughput (%) | 10.95 % | 10.92 % | -0.03 pp |
| Memory Throughput (%) | 78.25 % | 77.77 % | -0.48 pp |
| DRAM Throughput (%) | 5.43 % | 5.49 % | +0.06 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp |
| Registers/Thread | 96.00 register/thread | 80.00 register/thread | -16 |
| Waves/SM | 0.23 | 0.16 | -0.07 |
| No Eligible (%) | 87.26 % | 87.33 % | +0.07 pp |
| IPC Active | 0.51 inst/cycle | 0.50 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  30340 ns  (1.000×)

Final best: 30620 ns  (0.9909×)
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
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_23(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -67,10 +81,11 @@

         xindex = idx_n + 256*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 32*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        idx_k = b_k_idx_vals          # shape [BLOCK_K, 1]
+        idx_n = offs_b_n[None, :]     # shape [1, BLOCK_N]
+        # stride_bk=1 means K is contiguous; stride_bn=256 means N stride is 256
+        xindex = idx_k + 256 * idx_n  # shape [BLOCK_K, BLOCK_N], coalesced along K
+        b = tl.load(B + xindex)        # threads vary k (stride 1) => fully coalesced
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -88,8 +103,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
