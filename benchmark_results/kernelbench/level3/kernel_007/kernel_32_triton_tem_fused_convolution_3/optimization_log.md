# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55330 ns — **Best latency**: 55490 ns (**0.9971×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_32`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_32_triton_tem_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (93,1,1), Waves/SM: 0.18
- **Registers/thread**: 128
- **Duration**: 55330 ns
- **SM Throughput**: 15.43 % | **Memory**: 59.90 % | **DRAM**: 1.53 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 77.05 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotuned_tile_sizes | 0.943× | 0.943× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_reduction | 1.007× | 0.997× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_grid_coverage | 1.000× | 0.806× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 58660 ns | +3330 ns (+6.0%) | 55490 ns | +160 ns (+0.3%) | 68670 ns | +13340 ns (+24.1%) |
| SM Throughput (%) | 15.43 % | 14.75 % | -0.68 pp | 15.40 % | -0.03 pp | 12.06 % | -3.37 pp |
| Memory Throughput (%) | 59.90 % | 56.60 % | -3.30 pp | 59.80 % | -0.10 pp | 27.05 % | -32.85 pp |
| DRAM Throughput (%) | 1.53 % | 1.44 % | -0.09 pp | 1.53 % | +0.00 pp | 1.24 % | -0.29 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 168.00 register/thread | +40 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.12 | -0.06 |
| No Eligible (%) | 77.05 % | 78.20 % | +1.15 pp | 77.02 % | -0.03 pp | 62.08 % | -14.97 pp |
| IPC Active | 0.92 inst/cycle | 0.87 inst/cycle | -0.05 | 0.92 inst/cycle | +0.00 | 0.76 inst/cycle | -0.16 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_direct_b_load | 0.994× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 55330 ns | 55200 ns | -130 ns (-0.2%) |
| SM Throughput (%) | 15.43 % | 15.43 % | +0.00 pp |
| Memory Throughput (%) | 59.90 % | 59.91 % | +0.01 pp |
| DRAM Throughput (%) | 1.53 % | 1.53 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 |
| No Eligible (%) | 77.05 % | 77.02 % | -0.03 pp |
| IPC Active | 0.92 inst/cycle | 0.91 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_tensor_descriptor_api | 0.945× | 0.945× | ✓ |  |
| C | memory_coalescing_b_load_transpose_for_bank_conflict_elimination | 1.005× | 1.003× | ✓ |  |
| D | autotune_tile_sizes_as_constexpr_args_with_dynamic_grid | 0.995× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_ten... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 55330 ns | 58530 ns | +3200 ns (+5.8%) | 55170 ns | -160 ns (-0.3%) | 55070 ns | -260 ns (-0.5%) |
| SM Throughput (%) | 15.43 % | 14.76 % | -0.67 pp | 15.40 % | -0.03 pp | 15.38 % | -0.05 pp |
| Memory Throughput (%) | 59.90 % | 56.65 % | -3.25 pp | 59.81 % | -0.09 pp | 59.72 % | -0.18 pp |
| DRAM Throughput (%) | 1.53 % | 1.45 % | -0.08 pp | 1.54 % | +0.01 pp | 1.54 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.24 % | -0.01 pp | 6.24 % | -0.01 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 77.05 % | 78.22 % | +1.17 pp | 77.08 % | +0.03 pp | 77.05 % | +0.00 pp |
| IPC Active | 0.92 inst/cycle | 0.87 inst/cycle | -0.05 | 0.92 inst/cycle | +0.00 | 0.92 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  55330 ns  (1.000×)

Final best: 55490 ns  (0.9971×)
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
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_32(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -66,12 +80,12 @@

         xindex = idx_n + 480*idx_m
         a = tl.load(A + (xindex))
 
-        # Direct [BLOCK_K, BLOCK_N] load — no tl.trans, no shared memory bank conflicts
+        # Direct [BLOCK_K, BLOCK_N] load: eliminates tl.trans and shared memory bank conflicts
         k_base = k_idx * BLOCK_K
         b_idx_k = (k_base + tl.arange(0, BLOCK_K))[:, None]  # [BLOCK_K, 1]
         b_idx_n = offs_b_n[None, :]                           # [1, BLOCK_N]
         b_ptrs = B + b_idx_k * 1 + b_idx_n * 480             # [BLOCK_K, BLOCK_N]
-        b = tl.load(b_ptrs)                                   # shape [BLOCK_K, BLOCK_N], no tl.trans
+        b = tl.load(b_ptrs)                                   # already [BLOCK_K, BLOCK_N], no tl.trans needed
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -88,7 +102,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
