# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9540 ns — **Best latency**: 9540 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_14_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 54
- **Duration**: 9540 ns
- **SM Throughput**: 0.06 % | **Memory**: 0.50 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.984× | 1.004× | ✓ |  |
| D | autotune_tile_sizes_and_reduce_shared_memory_pressure | 0.996× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 9540 ns | 9500 ns | -40 ns (-0.4%) | 9570 ns | +30 ns (+0.3%) |
| SM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Memory Throughput (%) | 0.50 % | 0.51 % | +0.01 pp | 0.49 % | -0.01 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.08 % | +0.01 pp |
| Achieved Occupancy (%) | 6.10 % | 6.10 % | +0.00 pp | 6.10 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 54.00 register/thread | +0 | 38.00 register/thread | -16 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.13 % | 91.11 % | -0.02 pp | 91.76 % | +0.63 pp |
| IPC Active | 0.35 inst/cycle | 0.36 inst/cycle | +0.01 | 0.32 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_coalesced_global_load | 1.002× | 0.975× | ✓ | **✓ best** |
| C | fix_b_matrix_bank_conflicts_with_transposed_smem_staging | 0.995× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | fix_b_matrix_bank... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 9540 ns | 9790 ns | +250 ns (+2.6%) | 9540 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.06 % | +0.00 pp |
| Memory Throughput (%) | 0.50 % | 0.50 % | +0.00 pp | 0.50 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.08 % | +0.01 pp |
| Achieved Occupancy (%) | 6.10 % | 6.10 % | +0.00 pp | 6.10 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 54.00 register/thread | +0 | 38.00 register/thread | -16 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.13 % | 90.90 % | -0.23 pp | 92.01 % | +0.88 pp |
| IPC Active | 0.35 inst/cycle | 0.35 inst/cycle | +0.00 | 0.32 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  9540 ns  (1.000×)

Final best: 9540 ns  (1.0000×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14(in_ptr0, arg_A, arg_B, out_ptr0):
@@ -12,7 +13,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 16
     BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -60,18 +61,18 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 256*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 10*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # Fix B load: B has shape [10, 256] with strides (256, 1)
+        # B[n, k] = B_ptr + n*stride_bn + k*stride_bk = B_ptr + n*256 + k*1
+        k_offs = offs_k[:, None] + (k_idx * BLOCK_K)   # [BLOCK_K, 1]
+        n_offs = offs_b_n[None, :]                       # [1, BLOCK_N]
+        b = tl.load(B + k_offs * stride_bk + n_offs * stride_bn,
+                    mask=(k_offs < K) & (n_offs < N), other=0.0)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -88,8 +89,6 @@

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
