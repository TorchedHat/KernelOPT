# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 5890 ns (**1.0102×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_to_contiguous_load_indexing_with_small_tile_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_083/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.01 % | **Memory**: 41.49 % | **DRAM**: 20.79 %
- **Achieved Occupancy**: 54.93 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.247× | 0.944× | ✓ |  |
| B | autotune_tile_sizes_to_improve_occupancy_and_hide_l1tex_latency | 1.105× | 0.958× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_swap | 1.321× | 0.969× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.020× | 0.641× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 6300 ns | +350 ns (+5.9%) | 6210 ns | +260 ns (+4.4%) | 6140 ns | +190 ns (+3.2%) | 9280 ns | +3330 ns (+56.0%) |
| SM Throughput (%) | 16.01 % | 16.63 % | +0.62 pp | 16.72 % | +0.71 pp | 14.56 % | -1.45 pp | 27.09 % | +11.08 pp |
| Memory Throughput (%) | 41.49 % | 39.61 % | -1.88 pp | 40.65 % | -0.84 pp | 26.56 % | -14.93 pp | 48.51 % | +7.02 pp |
| DRAM Throughput (%) | 20.79 % | 19.52 % | -1.27 pp | 19.89 % | -0.90 pp | 5.02 % | -15.77 pp | 13.32 % | -7.47 pp |
| Achieved Occupancy (%) | 54.93 % | 56.19 % | +1.26 pp | 55.73 % | +0.80 pp | 41.98 % | -12.95 pp | 52.89 % | -2.04 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 24.00 register/thread | -2 | 40.00 register/thread | +14 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.99 | +0.25 |
| No Eligible (%) | 70.34 % | 70.92 % | +0.58 pp | 70.27 % | -0.07 pp | 68.39 % | -1.95 pp | 58.03 % | -12.31 pp |
| IPC Active | 1.15 inst/cycle | 1.09 inst/cycle | -0.06 | 1.13 inst/cycle | -0.02 | 1.21 inst/cycle | +0.06 | 1.72 inst/cycle | +0.57 |


### Key Learnings

#### ❌ Plan D: `autotune_tile_sizes_for_occupancy_and_coalescing` (0.641×)

Adding @triton.autotune with larger XBLOCK/YBLOCK configurations caused a 56% slowdown despite improving eligible warps per scheduler (0.96→1.60) and waves/SM (0.74→0.99). The winning autotuned config increased registers per thread from 26 to 40 and dynamic shared memory per block from 4.1 KB to 16.4 KB, which reduced theoretical occupancy from 100% to 75% (limited by registers) and more than doubled executed instructions (771K→1847K), resulting in a net 56% longer kernel duration. When the baseline kernel is already register-efficient and near-optimal in instruction count, introducing autotune configs with larger tiles can select a configuration that trades occupancy for tile size, increasing register pressure and instruction count enough to dominate any latency-hiding benefit.

> **Applicable when**: Kernel has very low waves/SM (<0.5), very low achieved occupancy (<40%), and the baseline register count is already high (>32 registers/thread) leaving room for larger tiles without occupancy loss.

> **Avoid when**: Baseline kernel already has low register pressure (≤26 registers/thread) and theoretical occupancy near 100%; in this case, autotuning larger tile configs will select configurations that increase register pressure, reduce theoretical occupancy, and multiply instruction count, causing net regression even if eligible warps per scheduler improves slightly.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_transpose_to_coalesce_strided_stores | 1.323× | 0.958× | ✓ | **✓ best** |
| B | vectorized_loads_with_max_contiguous_hints | 1.009× | 1.005× | ✓ |  |
| C | coalesced_store_via_flat_output_index | 1.114× | 0.842× | ✓ |  |
| D | vectorized_loads_with_register_pressure_reduction | 1.006× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ | vectorized_loads_... | Δ | coalesced_store_v... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 6210 ns | +260 ns (+4.4%) | 5920 ns | -30 ns (-0.5%) | 7070 ns | +1120 ns (+18.8%) | 6050 ns | +100 ns (+1.7%) |
| SM Throughput (%) | 16.01 % | 14.06 % | -1.95 pp | 16.21 % | +0.20 pp | 20.95 % | +4.94 pp | 16.94 % | +0.93 pp |
| Memory Throughput (%) | 41.49 % | 25.61 % | -15.88 pp | 42.40 % | +0.91 pp | 40.75 % | -0.74 pp | 38.25 % | -3.24 pp |
| DRAM Throughput (%) | 20.79 % | 4.97 % | -15.82 pp | 20.85 % | +0.06 pp | 5.82 % | -14.97 pp | 20.31 % | -0.48 pp |
| Achieved Occupancy (%) | 54.93 % | 43.83 % | -11.10 pp | 55.51 % | +0.58 pp | 55.71 % | +0.78 pp | 55.41 % | +0.48 pp |
| Registers/Thread | 26.00 register/thread | 24.00 register/thread | -2 | 26.00 register/thread | +0 | 24.00 register/thread | -2 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.34 % | 70.19 % | -0.15 pp | 70.98 % | +0.64 pp | 62.90 % | -7.44 pp | 70.90 % | +0.56 pp |
| IPC Active | 1.15 inst/cycle | 1.11 inst/cycle | -0.04 | 1.13 inst/cycle | -0.02 | 1.40 inst/cycle | +0.25 | 1.14 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan C: `coalesced_store_via_flat_output_index` (0.842×)

Restructuring a strided-store layout permutation kernel to use a flat output index (making stores stride-1) worsened performance by 16% despite improving achieved occupancy (42%→56%) and L1 hit rate (60%→70%). The flat-index approach introduced integer divide/modulo to recover channel and spatial indices on the load side, which caused the load access pattern to become non-contiguous: excessive sectors grew from 106K to 314K (52%→71% of total sectors) and executed instructions nearly doubled (618K→1019K). The net effect was that warp cycles per issued instruction increased (22.5→24.2) and kernel duration grew from 6.14 µs to 7.07 µs, because the ALU overhead of the integer division and the newly uncoalesced loads outweighed the benefit of coalesced stores.

> **Applicable when**: Never: this specific transformation (flat output index for store coalescing in a layout permutation) is not beneficial when it converts a contiguous load into a non-contiguous one and adds integer divide/modulo overhead.

> **Avoid when**: When the kernel performs a layout permutation where the input is already contiguous along the fast-varying thread dimension (stride-1 loads). Converting to a flat output index to coalesce stores will uncoalesce the loads, increase excessive sectors, and nearly double instruction count due to integer divide/modulo, causing net regression even if occupancy improves. The load uncoalescing penalty dominates the store coalescing benefit in short-duration memory-bound kernels.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_load_to_reduce_l1_thrash | 0.904× | 0.949× | ✓ |  |
| B | evict_first_load_policy_for_streaming_transpose | 0.892× | 0.939× | ✓ |  |
| C | evict_first_store_plus_vectorized_load_hints | 1.262× | 1.015× | ✓ |  |
| D | revert_to_contiguous_load_indexing_with_small_tile_autotune | 1.383× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | evict_first_load_... | Δ | evict_first_store... | Δ | revert_to_contigu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 6270 ns | +320 ns (+5.4%) | 6340 ns | +390 ns (+6.6%) | 5860 ns | -90 ns (-1.5%) | 5890 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 16.01 % | 16.23 % | +0.22 pp | 16.48 % | +0.47 pp | 14.82 % | -1.19 pp | 16.61 % | +0.60 pp |
| Memory Throughput (%) | 41.49 % | 39.91 % | -1.58 pp | 38.99 % | -2.50 pp | 26.99 % | -14.50 pp | 42.69 % | +1.20 pp |
| DRAM Throughput (%) | 20.79 % | 19.58 % | -1.21 pp | 19.47 % | -1.32 pp | 5.27 % | -15.52 pp | 20.97 % | +0.18 pp |
| Achieved Occupancy (%) | 54.93 % | 55.07 % | +0.14 pp | 55.56 % | +0.63 pp | 44.98 % | -9.95 pp | 55.14 % | +0.21 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 24.00 register/thread | -2 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.34 % | 70.92 % | +0.58 pp | 70.74 % | +0.40 pp | 66.93 % | -3.41 pp | 71.42 % | +1.08 pp |
| IPC Active | 1.15 inst/cycle | 1.13 inst/cycle | -0.02 | 1.09 inst/cycle | -0.06 | 1.21 inst/cycle | +0.06 | 1.14 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.0102× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_and_num_stages_on_small_tile_configs | 1.291× | 1.005× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 0.902× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 5920 ns | -30 ns (-0.5%) | 6140 ns | +190 ns (+3.2%) |
| SM Throughput (%) | 16.01 % | 13.88 % | -2.13 pp | 20.82 % | +4.81 pp |
| Memory Throughput (%) | 41.49 % | 25.31 % | -16.18 pp | 39.48 % | -2.01 pp |
| DRAM Throughput (%) | 20.79 % | 5.21 % | -15.58 pp | 20.05 % | -0.74 pp |
| Achieved Occupancy (%) | 54.93 % | 45.34 % | -9.59 pp | 74.15 % | +19.22 pp |
| Registers/Thread | 26.00 register/thread | 24.00 register/thread | -2 | 20.00 register/thread | -6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.34 % | 69.08 % | -1.26 pp | 66.94 % | -3.40 pp |
| IPC Active | 1.15 inst/cycle | 1.23 inst/cycle | +0.08 | 1.31 inst/cycle | +0.16 |

---

## Iteration 5

**Best so far**: 1.0102× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_grid_stride_loop | 0.453× | 0.387× | ✓ |  |
| D | eliminate_partial_wave_via_grid_alignment | 0.887× | 0.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 15390 ns | +9440 ns (+158.7%) | 6140 ns | +190 ns (+3.2%) |
| SM Throughput (%) | 16.01 % | 23.78 % | +7.77 pp | 20.37 % | +4.36 pp |
| Memory Throughput (%) | 41.49 % | 65.77 % | +24.28 pp | 38.94 % | -2.55 pp |
| DRAM Throughput (%) | 20.79 % | 7.99 % | -12.80 pp | 20.12 % | -0.67 pp |
| Achieved Occupancy (%) | 54.93 % | 60.94 % | +6.01 pp | 72.81 % | +17.88 pp |
| Registers/Thread | 26.00 register/thread | 40.00 register/thread | +14 | 20.00 register/thread | -6 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 1.48 | +0.74 |
| No Eligible (%) | 70.34 % | 70.51 % | +0.17 pp | 67.88 % | -2.46 pp |
| IPC Active | 1.15 inst/cycle | 1.17 inst/cycle | +0.02 | 1.33 inst/cycle | +0.18 |


### Key Learnings

#### ❌ Plan B: `persistent_kernel_grid_stride_loop` (0.387×)

Converting a 2D multi-CTA launch into a persistent 1D grid-stride loop (132 CTAs, one per SM) caused a 2.6x slowdown despite improving waves/SM (0.74→0.99) and achieved occupancy (45%→61%). The while-loop control flow and per-iteration tile index arithmetic (integer divide/modulo of tile_id) inflated registers per thread from 24 to 40, reducing theoretical occupancy from 100% to 75% (register-limited), and multiplied executed instructions by 4.6x (618K→2816K). The net effect was that warp cycles per issued instruction increased from 22.4 to 33.1 and kernel duration nearly tripled (5.92 µs→15.39 µs). For very short kernels (<10 µs) with modest wave counts (~0.74), the overhead of the grid-stride loop body (index recomputation, branch, register pressure) far exceeds the benefit of eliminating the partial-wave tail.

> **Applicable when**: Never beneficial for very short kernels (<10 µs) with low register counts in the baseline. The grid-stride loop overhead dominates any wave-tail savings.

> **Avoid when**: When the baseline kernel is short (<10 µs), has low register pressure (≤24 registers/thread), and theoretical occupancy near 100%. The while-loop and per-iteration integer divide/modulo for tile index decomposition inflate register count (24→40), reduce theoretical occupancy (100%→75%), and multiply instruction count (4.6x), causing a net 2.6x slowdown. The partial-wave tail savings (~0.26 waves) are negligible compared to the instruction bloat. Also avoid when the grid is already close to 1 wave (0.74 waves) — the tail is small and not worth the loop overhead.

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 3     5890 ns  (1.010×)  — revert_to_contiguous_load_indexing_with_small_tile_autotune

Final best: 5890 ns  (1.0102×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=1),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +27,14 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask)
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

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
