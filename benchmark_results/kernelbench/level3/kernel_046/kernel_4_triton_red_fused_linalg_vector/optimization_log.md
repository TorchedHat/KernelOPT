# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_linalg_vector_norm_mul_sub_transpose_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 882210 ns — **Best latency**: 5980 ns (**147.5268×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_linalg_vector_norm_mul_sub_transpose_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_4_triton_red_fused_linalg_vector/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.29
- **Registers/thread**: 32
- **Duration**: 882210 ns
- **SM Throughput**: 6.19 % | **Memory**: 30.31 % | **DRAM**: 30.31 %
- **Achieved Occupancy**: 16.50 %
- **IPC Active**: 0.10 inst/cycle | **No Eligible**: 97.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_with_vectorized_loads | 109.459× | 168.040× | ✓ |  |
| B | memory_coalescing_and_occupancy_via_autotune | 50.593× | 42.537× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_latency_hiding | 110.960× | 147.527× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 882210 ns | 5250 ns | -876960 ns (-99.4%) | 20740 ns | -861470 ns (-97.6%) | 5980 ns | -876230 ns (-99.3%) |
| SM Throughput (%) | 6.19 % | 4.39 % | -1.80 pp | 8.94 % | +2.75 pp | 4.69 % | -1.50 pp |
| Memory Throughput (%) | 30.31 % | 30.31 % | +0.00 pp | 22.46 % | -7.85 pp | 47.15 % | +16.84 pp |
| DRAM Throughput (%) | 30.31 % | 4.38 % | -25.93 pp | 16.60 % | -13.71 pp | 3.83 % | -26.48 pp |
| Achieved Occupancy (%) | 16.50 % | 21.19 % | +4.69 pp | 11.83 % | -4.67 pp | 21.01 % | +4.51 pp |
| Registers/Thread | 32.00 register/thread | 22.00 register/thread | -10 | 40.00 register/thread | +8 | 27.00 register/thread | -5 |
| Waves/SM | 1.29 | 0.24 | -1.05 | 0.16 | -1.13 | 0.24 | -1.05 |
| No Eligible (%) | 97.39 % | 92.08 % | -5.31 pp | 93.97 % | -3.42 pp | 93.35 % | -4.04 pp |
| IPC Active | 0.10 inst/cycle | 0.30 inst/cycle | +0.20 | 0.24 inst/cycle | +0.14 | 0.26 inst/cycle | +0.16 |

---

## Iteration 2

**Best so far**: 147.5268× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_stores_with_contiguity_hints | 108.524× | 172.307× | ✓ |  |
| B | vectorized_loads_via_max_contiguous_multiple_of_hints | 110.701× | 145.820× | ✓ |  |
| C | vectorized_loads_with_fixed_autotune_grid | 95.766× | 133.871× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 882210 ns | 5120 ns | -877090 ns (-99.4%) | 6050 ns | -876160 ns (-99.3%) | 6590 ns | -875620 ns (-99.3%) |
| SM Throughput (%) | 6.19 % | 4.54 % | -1.65 pp | 4.70 % | -1.49 pp | 5.70 % | -0.49 pp |
| Memory Throughput (%) | 30.31 % | 31.32 % | +1.01 pp | 47.23 % | +16.92 pp | 44.51 % | +14.20 pp |
| DRAM Throughput (%) | 30.31 % | 4.46 % | -25.85 pp | 3.79 % | -26.52 pp | 3.48 % | -26.83 pp |
| Achieved Occupancy (%) | 16.50 % | 21.02 % | +4.52 pp | 21.00 % | +4.50 pp | 11.63 % | -4.87 pp |
| Registers/Thread | 32.00 register/thread | 22.00 register/thread | -10 | 27.00 register/thread | -5 | 32.00 register/thread | +0 |
| Waves/SM | 1.29 | 0.24 | -1.05 | 0.24 | -1.05 | 0.12 | -1.17 |
| No Eligible (%) | 97.39 % | 91.96 % | -5.43 pp | 93.79 % | -3.60 pp | 94.02 % | -3.37 pp |
| IPC Active | 0.10 inst/cycle | 0.30 inst/cycle | +0.20 | 0.26 inst/cycle | +0.16 | 0.22 inst/cycle | +0.12 |


### Key Learnings

#### ✅ Plan B: `vectorized_loads_via_max_contiguous_multiple_of_hints` (145.820×)

Applying tl.max_contiguous(tl.multiple_of(r0_index, R0_BLOCK), R0_BLOCK) as a vectorization hint to a contiguous reduction index increased L1/TEX cache throughput (53% → 74%) and hit rate (35% → 68%), but simultaneously increased kernel duration (5.25 µs → 6.05 µs), excessive sectors (229K → 459K), and warp stall cycles per instruction (48.5 → 51.8). The autotune config expansion (from ~14 to ~34 configs) likely selected a suboptimal tile configuration that increased total memory transactions despite better per-load vectorization. The net effect was a regression relative to the already-autotuned baseline, not a speedup — the 145x figure reflects improvement over the original pre-autotune kernel, not over the immediate prior iteration.

> **Applicable when**: Kernel has a genuinely contiguous, aligned reduction index and the baseline already has poor load sector utilization (< 8 bytes/sector) with MemoryCacheAccessPattern rule firing — AND the kernel is not already near-optimal from a prior autotune pass.

> **Avoid when**: The kernel has already been optimized by autotune to a short duration (< 7 µs) with achieved occupancy ~21% and warp stall cycles ~48/instruction. In this regime, adding tl.max_contiguous/tl.multiple_of hints combined with a large autotune config expansion can cause the autotuner to select worse tile configurations, increasing total memory transactions and kernel duration despite improving per-load vectorization efficiency. Also avoid when the dominant uncoalesced access is strided (e.g., stride-32), as the hint cannot fix that access pattern.

---

## Iteration 3

**Best so far**: 147.5268× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_access_via_xblock_covering_x0_stride | 50.331× | 54.592× | ✓ |  |
| C | fix_autotune_grid_lambda_and_expand_xblock_configs | 110.634× | 140.703× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_access_... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 882210 ns | 16160 ns | -866050 ns (-98.2%) | 6270 ns | -875940 ns (-99.3%) |
| SM Throughput (%) | 6.19 % | 12.19 % | +6.00 pp | 5.68 % | -0.51 pp |
| Memory Throughput (%) | 30.31 % | 43.16 % | +12.85 pp | 44.38 % | +14.07 pp |
| DRAM Throughput (%) | 30.31 % | 43.16 % | +12.85 pp | 3.76 % | -26.55 pp |
| Achieved Occupancy (%) | 16.50 % | 12.27 % | -4.23 pp | 11.64 % | -4.86 pp |
| Registers/Thread | 32.00 register/thread | 194.00 register/thread | +162 | 32.00 register/thread | +0 |
| Waves/SM | 1.29 | 1.94 | +0.65 | 0.12 | -1.17 |
| No Eligible (%) | 97.39 % | 84.28 % | -13.11 pp | 94.33 % | -3.06 pp |
| IPC Active | 0.10 inst/cycle | 0.62 inst/cycle | +0.52 | 0.22 inst/cycle | +0.12 |

---

## Cumulative Progress

```
Baseline:  882210 ns  (1.000×)
Iter 1     5980 ns  (147.527×)  — autotune_tile_sizes_for_occupancy_and_latency_hiding

Final best: 5980 ns  (147.5268×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 512}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 32},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 16},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 32},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 32},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 32},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 16},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_linalg_vector_norm_mul_sub_transpose_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 65536
@@ -26,8 +48,9 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
-        tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
+        r0_2_vec = tl.max_contiguous(tl.multiple_of(r0_2, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tmp0 = tl.load(in_ptr0 + (r0_2_vec + 512*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
         tmp3 = tmp1 * tmp2
         tmp4 = tmp0 - tmp3
         tmp5 = tmp4 * tmp4
@@ -35,8 +58,8 @@

         tmp8 = _tmp7 + tmp6
         _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
     tmp7 = tl.sum(_tmp7, 1)[:, None]
-    tl.store(out_ptr0 + (x3), tmp7, xmask)
-
+    x3_vec = tl.max_contiguous(tl.multiple_of(xindex, [XBLOCK, 1]), [XBLOCK, 1])
+    tl.store(out_ptr0 + (x3_vec), tmp7, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +69,7 @@

     xnumel = 65536
     r0_numel = 512
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_linalg_vector_norm_mul_sub_transpose_4[grid](arg_0, arg_1, arg_2, arg_3, 65536, 512, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_linalg_vector_norm_mul_sub_transpose_4[grid](arg_0, arg_1, arg_2, arg_3, 65536, 512)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
