# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_gelu_t_view_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 185700 ns — **Best latency**: 186620 ns (**0.9951×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_gelu_t_view_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_5_triton_tem_fused_addmm_gelu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.93
- **Registers/thread**: 255
- **Duration**: 185700 ns
- **SM Throughput**: 37.19 % | **Memory**: 54.07 % | **DRAM**: 5.83 %
- **Achieved Occupancy**: 11.82 %
- **IPC Active**: 1.59 inst/cycle | **No Eligible**: 60.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.998× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 185700 ns | 186300 ns | +600 ns (+0.3%) |
| SM Throughput (%) | 37.19 % | 37.26 % | +0.07 pp |
| Memory Throughput (%) | 54.07 % | 54.19 % | +0.12 pp |
| DRAM Throughput (%) | 5.83 % | 5.81 % | -0.02 pp |
| Achieved Occupancy (%) | 11.82 % | 11.82 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 |
| No Eligible (%) | 60.25 % | 60.19 % | -0.06 pp |
| IPC Active | 1.59 inst/cycle | 1.59 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policies_for_streaming_a_reused_b | 1.000× | 0.994× | ✓ | **✓ best** |
| D | fix_b_load_indexing_and_increase_block_k_for_register_relief | 0.169× | 0.164× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | fix_b_load_indexi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 185700 ns | 186780 ns | +1080 ns (+0.6%) | 1130000 ns | +944300 ns (+508.5%) |
| SM Throughput (%) | 37.19 % | 37.24 % | +0.05 pp | 7.45 % | -29.74 pp |
| Memory Throughput (%) | 54.07 % | 54.15 % | +0.08 pp | 15.24 % | -38.83 pp |
| DRAM Throughput (%) | 5.83 % | 5.79 % | -0.04 pp | 11.30 % | +5.47 pp |
| Achieved Occupancy (%) | 11.82 % | 11.82 % | +0.00 pp | 6.25 % | -5.57 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 1.86 | +0.93 |
| No Eligible (%) | 60.25 % | 60.16 % | -0.09 pp | 91.52 % | +31.27 pp |
| IPC Active | 1.59 inst/cycle | 1.59 inst/cycle | +0.00 | 0.34 inst/cycle | -1.25 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | tma_loads_via_block_ptr_with_block_k_32 | 1.007× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 185700 ns | 185250 ns | -450 ns (-0.2%) |
| SM Throughput (%) | 37.19 % | 36.77 % | -0.42 pp |
| Memory Throughput (%) | 54.07 % | 53.87 % | -0.20 pp |
| DRAM Throughput (%) | 5.83 % | 5.70 % | -0.13 pp |
| Achieved Occupancy (%) | 11.82 % | 11.87 % | +0.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 |
| No Eligible (%) | 60.25 % | 59.62 % | -0.63 pp |
| IPC Active | 1.59 inst/cycle | 1.62 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.0024× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_async_pipeline_via_num_stages | 1.007× | 1.002× | ✓ |  |
| C | eviction_policies_streaming_a_reused_b_via_cache_modifier | 1.010× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_async_pipelin... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 185700 ns | 185380 ns | -320 ns (-0.2%) | 186620 ns | +920 ns (+0.5%) |
| SM Throughput (%) | 37.19 % | 36.90 % | -0.29 pp | 36.99 % | -0.20 pp |
| Memory Throughput (%) | 54.07 % | 54.04 % | -0.03 pp | 54.16 % | +0.09 pp |
| DRAM Throughput (%) | 5.83 % | 5.70 % | -0.13 pp | 5.64 % | -0.19 pp |
| Achieved Occupancy (%) | 11.82 % | 11.87 % | +0.05 pp | 11.88 % | +0.06 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 60.25 % | 59.67 % | -0.58 pp | 59.69 % | -0.56 pp |
| IPC Active | 1.59 inst/cycle | 1.61 inst/cycle | +0.02 | 1.61 inst/cycle | +0.02 |

---

## Iteration 5

**Best so far**: 1.0024× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  185700 ns  (1.000×)
Iter 3     185250 ns  (1.002×)  — tma_loads_via_block_ptr_with_block_k_32

Final best: 186620 ns  (0.9951×)
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
 def triton_tem_fused_addmm_gelu_t_view_5(arg_A, arg_B, out_ptr0):
@@ -44,37 +45,37 @@

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
 
+    # Construct block pointers before the k-loop (TMA descriptors on Hopper)
+    # A has shape (M=31360, K=384) with strides (384, 1) - row-major
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    # B is stored as [N=96, K=384] with strides (384, 1)
+    # Viewed as [K=384, N=96] for matmul: strides=(stride_bk=1, stride_bn=384)
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 384*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
+        # A is streaming (used once per CTA): .cg caches at L2 only, bypasses L1
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero', cache_modifier='.cg')
+        # B is reused across all 245 M-tiles: .ca caches at all levels (evict_last / pin in L2)
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero', cache_modifier='.ca')
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +87,6 @@

     # inductor generates a suffix
     xindex = idx_n + 96*idx_m
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
