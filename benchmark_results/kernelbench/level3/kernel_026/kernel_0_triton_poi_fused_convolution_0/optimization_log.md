# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 3260 ns (**1.7975×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_via_correct_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_026/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 16.61 % | **Memory**: 41.48 % | **DRAM**: 21.12 %
- **Achieved Occupancy**: 55.62 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 70.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.451× | 1.798× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_underutilized_grid | 1.149× | 0.435× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3260 ns | -2600 ns (-44.4%) | 13470 ns | +7610 ns (+129.9%) |
| SM Throughput (%) | 16.61 % | 7.29 % | -9.32 pp | 48.63 % | +32.02 pp |
| Memory Throughput (%) | 41.48 % | 21.24 % | -20.24 pp | 59.99 % | +18.51 pp |
| DRAM Throughput (%) | 21.12 % | 5.09 % | -16.03 pp | 9.12 % | -12.00 pp |
| Achieved Occupancy (%) | 55.62 % | 27.06 % | -28.56 pp | 63.83 % | +8.21 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 1.98 | +1.24 |
| No Eligible (%) | 70.67 % | 74.38 % | +3.71 pp | 44.51 % | -26.16 pp |
| IPC Active | 1.08 inst/cycle | 0.89 inst/cycle | -0.19 | 2.19 inst/cycle | +1.11 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.798×)

When a Triton kernel has high L1TEX scoreboard stall fraction (>40% of warp cycles) combined with uncoalesced global stores (>15% excess sectors), the fixed tile sizes chosen by the compiler may be suboptimal. Adding a @triton.autotune decorator with a sweep of XBLOCK/YBLOCK configurations and num_warps values allows the runtime to empirically select the tile shape that best balances store coalescing and wave count. In this case, the autotuner reduced warp cycles per issued instruction from 29.6 to 18.0 and improved L2 hit rate from 60.5% to 77.8%, yielding a 1.8x wall-time speedup despite lower achieved occupancy, because fewer total instructions were needed per element.

> **Applicable when**: Triton kernel with: (1) L1TEX scoreboard stall fraction >40% of warp cycles, (2) uncoalesced global store excess sectors >15%, (3) fixed hardcoded XBLOCK/YBLOCK tile sizes that were never autotuned, (4) 2D grid with both spatial and channel/batch dimensions. Especially effective for layout permutation kernels (e.g. NCHW→NHWC) where strided stores cause cache thrashing.

> **Avoid when**: Kernel is already autotuned or the tile sizes are constrained by shared memory layout that must match the tile shape. Also avoid if the kernel is so small that autotune overhead dominates (grid_size < ~100 CTAs). Do not apply if the bottleneck is compute-bound rather than memory-latency-bound.

#### ❌ Plan D: `autotune_tile_sizes_for_underutilized_grid` (0.418×)

Replacing autotune configs that used num_stages=1 with configs using num_stages=3 caused the selected config to allocate ~32 KB of dynamic shared memory per block (up from 512 bytes), collapsing Block Limit Shared Mem from 42 to 6 blocks/SM and capping Theoretical Occupancy at 75% instead of 100%. Although waves_per_sm improved from 0.37 to 1.98 and L1/TEX hit rate rose from 0% to 49%, elapsed cycles increased 4.3x (4835 → 20961 cycles), yielding a 0.42x regression. For simple elementwise/transpose kernels with no explicit shared memory reuse, num_stages>1 allocates hidden pipeline buffers that destroy occupancy without providing any cache reuse benefit.

> **Applicable when**: Never applicable as described — this is a negative example. The pattern of autotuning with num_stages>1 on kernels with no shared memory reuse causes regression.

> **Avoid when**: Do NOT add num_stages>1 to @triton.autotune configs for kernels that perform simple elementwise or transpose operations with no explicit shared memory tiling. The software pipelining buffer allocated by num_stages=3 consumes large amounts of shared memory (~tile_size * num_stages * dtype_bytes per block), collapsing Block Limit Shared Mem and theoretical occupancy even when the kernel body never explicitly uses shared memory. This regression pattern has now been observed twice (iterations 1 and 2) with consistent metrics: Block Limit Shared Mem drops from 26-42 to 6, Theoretical Occupancy drops from 100% to 75%, and elapsed cycles increase 2-4x. Prefer num_stages=1 for such kernels.

---

## Iteration 2

**Best so far**: 1.7975× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_correct_contiguous_hints | 1.451× | 1.798× | ✓ |  |
| B | shared_memory_transpose_to_eliminate_uncoalesced_stores | 1.295× | 1.605× | ✓ |  |
| C | vectorized_loads_via_yblock1_configs_and_partial_wave_elimination | 1.434× | 0.418× | ✓ |  |
| D | fix_shared_memory_collapse_from_num_stages_pipelining | 1.450× | 1.429× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_tra... | Δ | vectorized_loads_... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3260 ns | -2600 ns (-44.4%) | 3650 ns | -2210 ns (-37.7%) | 14020 ns | +8160 ns (+139.2%) | 4100 ns | -1760 ns (-30.0%) |
| SM Throughput (%) | 16.61 % | 7.73 % | -8.88 pp | 18.87 % | +2.26 pp | 50.18 % | +33.57 pp | 26.65 % | +10.04 pp |
| Memory Throughput (%) | 41.48 % | 21.24 % | -20.24 pp | 24.34 % | -17.14 pp | 61.95 % | +20.47 pp | 21.16 % | -20.32 pp |
| DRAM Throughput (%) | 21.12 % | 5.08 % | -16.04 pp | 4.55 % | -16.57 pp | 8.78 % | -12.34 pp | 4.08 % | -17.04 pp |
| Achieved Occupancy (%) | 55.62 % | 26.79 % | -28.83 pp | 54.51 % | -1.11 pp | 63.91 % | +8.29 pp | 70.57 % | +14.95 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 21.00 register/thread | -5 | 32.00 register/thread | +6 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.74 | +0.00 | 1.98 | +1.24 | 1.48 | +0.74 |
| No Eligible (%) | 70.67 % | 74.83 % | +4.16 pp | 60.66 % | -10.01 pp | 44.84 % | -25.83 pp | 36.98 % | -33.69 pp |
| IPC Active | 1.08 inst/cycle | 0.87 inst/cycle | -0.21 | 1.52 inst/cycle | +0.44 | 2.20 inst/cycle | +1.12 | 2.26 inst/cycle | +1.18 |


### Key Learnings

#### ⚠️ Plan A: `vectorized_loads_via_correct_contiguous_hints` (1.798×)

Applying tl.multiple_of and tl.max_contiguous to the 1D arange result before the [None,:] broadcast reshape — the correct API usage — produced no measurable improvement: duration remained 3.26 us, elapsed cycles were essentially unchanged (4835 vs 4849), and all memory/scheduler metrics were flat. The uncoalesced store bottleneck (50176 excess sectors, stride-3 writes) dominates and cannot be fixed by load-side vectorization hints alone. When the store-side access pattern is the primary bottleneck (stride > 1 in the warp-fast dimension), contiguous-load hints on the read side yield negligible benefit.

> **Applicable when**: Never — this strategy produced no speedup even when implemented correctly. The correct API form (1D hints before broadcast) is syntactically valid but does not help when the dominant bottleneck is uncoalesced stores rather than uncoalesced loads.

> **Avoid when**: Do not apply tl.max_contiguous / tl.multiple_of load hints when the kernel's primary bottleneck is uncoalesced global stores (UncoalescedGlobalAccess excess sectors >20%, MemoryCacheAccessPattern bytes/sector <12 on the store side). Load-side vectorization hints cannot compensate for store-side stride amplification. Also do not apply when the kernel is already autotuned and the load path is already near-optimal (L2 hit rate >75%, memory throughput near peak for the working set).

---

## Iteration 3

**Best so far**: 1.7975× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 1     3260 ns  (1.798×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3260 ns  (1.7975×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1},  num_warps=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2},  num_warps=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +43,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
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
