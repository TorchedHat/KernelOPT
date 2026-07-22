# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 5950 ns (**1.0168×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_021/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 17.16 % | **Memory**: 41.19 % | **DRAM**: 20.40 %
- **Achieved Occupancy**: 54.92 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.402× | 1.512× | ✓ | **✓ best** |
| C | memory_coalescing_swap_tile_dimensions | 1.172× | 1.490× | ✓ |  |
| D | add_autotune_tile_sizes_and_warps | 1.082× | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | add_autotune_tile... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4000 ns | -2050 ns (-33.9%) | 4060 ns | -1990 ns (-32.9%) | 5890 ns | -160 ns (-2.6%) |
| SM Throughput (%) | 17.16 % | 28.13 % | +10.97 pp | 16.26 % | -0.90 pp | 16.80 % | -0.36 pp |
| Memory Throughput (%) | 41.19 % | 22.37 % | -18.82 pp | 33.54 % | -7.65 pp | 42.69 % | +1.50 pp |
| DRAM Throughput (%) | 20.40 % | 4.12 % | -16.28 pp | 16.14 % | -4.26 pp | 21.06 % | +0.66 pp |
| Achieved Occupancy (%) | 54.92 % | 65.72 % | +10.80 pp | 53.71 % | -1.21 pp | 54.72 % | -0.20 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 20.00 register/thread | -6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.47 % | 39.09 % | -31.38 pp | 65.65 % | -4.82 pp | 70.87 % | +0.40 pp |
| IPC Active | 1.15 inst/cycle | 2.34 inst/cycle | +1.19 | 1.29 inst/cycle | +0.14 | 1.18 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.5125× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_eliminate_partial_wave | 1.410× | 1.017× | ✓ | **✓ best** |
| B | vectorized_loads_contiguous_hints_with_larger_xblock | 1.294× | 1.454× | ✓ |  |
| C | swap_arange_axes_and_vectorized_load_hints | 1.173× | 1.100× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | vectorized_loads_... | Δ | swap_arange_axes_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 5950 ns | -100 ns (-1.7%) | 4160 ns | -1890 ns (-31.2%) | 5500 ns | -550 ns (-9.1%) |
| SM Throughput (%) | 17.16 % | 27.98 % | +10.82 pp | 26.24 % | +9.08 pp | 19.15 % | +1.99 pp |
| Memory Throughput (%) | 41.19 % | 30.60 % | -10.59 pp | 20.81 % | -20.38 pp | 38.10 % | -3.09 pp |
| DRAM Throughput (%) | 20.40 % | 2.78 % | -17.62 pp | 3.98 % | -16.42 pp | 22.45 % | +2.05 pp |
| Achieved Occupancy (%) | 54.92 % | 73.22 % | +18.30 pp | 68.15 % | +13.23 pp | 52.18 % | -2.74 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 18.00 register/thread | -8 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.47 % | 39.45 % | -31.02 pp | 37.43 % | -33.04 pp | 65.89 % | -4.58 pp |
| IPC Active | 1.15 inst/cycle | 2.44 inst/cycle | +1.29 | 2.38 inst/cycle | +1.23 | 1.30 inst/cycle | +0.15 |

---

## Iteration 3

**Best so far**: 1.5125× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_arange_axes_to_coalesce_stride1_stores | 1.306× | 1.420× | ✓ |  |
| B | flatten_y_into_x_with_streaming_eviction_for_coalesced_stores | 1.325× | 1.444× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_arange_axes_... | Δ | flatten_y_into_x_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4260 ns | -1790 ns (-29.6%) | 4190 ns | -1860 ns (-30.7%) |
| SM Throughput (%) | 17.16 % | 26.59 % | +9.43 pp | 28.13 % | +10.97 pp |
| Memory Throughput (%) | 41.19 % | 21.08 % | -20.11 pp | 22.27 % | -18.92 pp |
| DRAM Throughput (%) | 20.40 % | 3.86 % | -16.54 pp | 3.93 % | -16.47 pp |
| Achieved Occupancy (%) | 54.92 % | 68.59 % | +13.67 pp | 68.20 % | +13.28 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.47 % | 37.04 % | -33.43 pp | 37.28 % | -33.19 pp |
| IPC Active | 1.15 inst/cycle | 2.41 inst/cycle | +1.26 | 2.41 inst/cycle | +1.26 |


### Key Learnings

#### ✅ Plan A: `swap_arange_axes_to_coalesce_stride1_stores` (1.490×)

In a 2D Triton kernel with YBLOCK×XBLOCK tiling, the dimension placed on `tl.arange(0, N)[None, :]` (the column/fast axis) is warp-parallel. If the output tensor's stride-1 dimension is currently on the slow axis (`[:, None]`), consecutive threads write with a stride > 1, causing uncoalesced stores (NCU UncoalescedGlobalAccess, low bytes-per-sector). Swapping the arange axes—putting the stride-1 output dimension on `[None, :]` and the strided dimension on `[:, None]`—makes consecutive threads write to consecutive addresses, reducing excessive sectors and L1TEX stall cycles. This swap also reduces register pressure (26→20 registers/thread) and shared memory usage, yielding a 1.49× speedup even though the uncoalesced-access metric paradoxically worsened in the after profile (the store pattern changed but the load pattern became the new bottleneck).

> **Applicable when**: Triton 2D kernel where the output tensor's stride-1 dimension is mapped to the slow arange axis ([:, None]) while a strided dimension is on the fast axis ([None, :]); NCU reports UncoalescedGlobalAccess with >10% excessive sectors and MemoryCacheAccessPattern showing <32 bytes/sector for stores; CPIStall on L1TEX is a top stall reason (>10 cycles/warp).

> **Avoid when**: The load pattern is the primary bottleneck and swapping axes would uncoalesce loads while coalescing stores (net negative); when both dimensions have the same stride; when the kernel is compute-bound rather than memory-latency-bound; when the swap would break correctness due to data dependencies across the tile.

---

## Iteration 4

**Best so far**: 1.5125× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staged_transpose_for_coalesced_stores | 1.218× | 1.321× | ✓ |  |
| C | shared_memory_staged_transpose_for_coalesced_stores | 1.290× | 1.501× | ✓ |  |
| D | add_triton_autotune | 0.931× | 1.112× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | shared_memory_sta... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4580 ns | -1470 ns (-24.3%) | 4030 ns | -2020 ns (-33.4%) | 5440 ns | -610 ns (-10.1%) |
| SM Throughput (%) | 17.16 % | 24.01 % | +6.85 pp | 25.85 % | +8.69 pp | 19.23 % | +2.07 pp |
| Memory Throughput (%) | 41.19 % | 21.20 % | -19.99 pp | 20.50 % | -20.69 pp | 38.76 % | -2.43 pp |
| DRAM Throughput (%) | 20.40 % | 6.76 % | -13.64 pp | 4.08 % | -16.32 pp | 22.75 % | +2.35 pp |
| Achieved Occupancy (%) | 54.92 % | 52.82 % | -2.10 pp | 67.49 % | +12.57 pp | 52.02 % | -2.90 pp |
| Registers/Thread | 26.00 register/thread | 28.00 register/thread | +2 | 18.00 register/thread | -8 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.47 % | 40.81 % | -29.66 pp | 37.43 % | -33.04 pp | 65.78 % | -4.69 pp |
| IPC Active | 1.15 inst/cycle | 2.41 inst/cycle | +1.26 | 2.41 inst/cycle | +1.26 | 1.34 inst/cycle | +0.19 |

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 1     4000 ns  (1.512×)  — vectorized_loads_and_autotune_tile_sizes

Final best: 5950 ns  (1.0168×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # NEW: YBLOCK=1 — y0 becomes a scalar per CTA, store stride resolved at compile time
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        # Existing configs retained
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
     xnumel = 50176
+    xoffset = tl.program_id(0) * XBLOCK
+    xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xrange[:, None]
+    xmask = xindex < xnumel
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +49,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
