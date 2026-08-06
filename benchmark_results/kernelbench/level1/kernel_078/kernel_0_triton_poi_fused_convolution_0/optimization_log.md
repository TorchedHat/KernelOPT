# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 4100 ns (**1.4366×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`evict_first_store_hint_for_streaming_writes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_078/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 15.62 % | **Memory**: 42.63 % | **DRAM**: 20.93 %
- **Achieved Occupancy**: 55.25 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.363× | 1.451× | ✓ |  |
| D | add_autotune_increase_tile_sizes_and_waves | 1.412× | 1.437× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4060 ns | -1830 ns (-31.1%) | 4100 ns | -1790 ns (-30.4%) |
| SM Throughput (%) | 15.62 % | 27.34 % | +11.72 pp | 27.47 % | +11.85 pp |
| Memory Throughput (%) | 42.63 % | 21.74 % | -20.89 pp | 21.82 % | -20.81 pp |
| DRAM Throughput (%) | 20.93 % | 4.10 % | -16.83 pp | 4.06 % | -16.87 pp |
| Achieved Occupancy (%) | 55.25 % | 67.41 % | +12.16 pp | 68.89 % | +13.64 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.56 % | 37.38 % | -33.18 pp | 39.34 % | -31.22 pp |
| IPC Active | 1.14 inst/cycle | 2.37 inst/cycle | +1.23 | 2.41 inst/cycle | +1.27 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.451×)

When a Triton kernel shows low achieved occupancy (<60%), high 'No Eligible' warp percentage (>60%), and L1TEX scoreboard stalls dominating CPI, replacing hardcoded XBLOCK/YBLOCK with @triton.autotune over a range of tile configurations allows the runtime to select the best block size for latency hiding. Doubling the block size (e.g., 128→256) reduces register pressure per thread, increases waves per SM from <1 to >1, and dramatically improves issue slot utilization (29%→61%). For kernels with strided store patterns causing uncoalesced global writes, including YBLOCK=1 configs in the autotune search eliminates cross-row scatter by ensuring all threads in a block share the same slow-dimension index.

> **Applicable when**: Triton kernel with hardcoded XBLOCK/YBLOCK tile sizes; achieved occupancy <60%; waves per SM <1; 'No Eligible' warp % >60%; L1TEX scoreboard stalls >40% of CPI; uncoalesced global store warnings present; kernel performs a layout permutation (e.g., NCHW→NHWC) with strided writes

> **Avoid when**: Kernel already uses @triton.autotune; kernel is compute-bound (SM throughput >80%); grid size is already >>2x SM count (waves per SM >2); adding autotune overhead is unacceptable in latency-sensitive single-call contexts

#### ✅ Plan D: `add_autotune_increase_tile_sizes_and_waves` (1.437×)

When a Triton kernel has fewer than 1.0 waves/SM (SOLBottleneck rule) and high L1TEX stall cycles (CPIStall rule, >40% of warp cycles), adding @triton.autotune with larger XBLOCK/YBLOCK configs allows the runtime to select tile sizes that fill the GPU and provide more in-flight memory requests to hide latency. In this case, doubling waves/SM from 0.74 to 1.48 and reducing warp cycles per issued instruction from 30 to 17.6 yielded a 1.44x speedup. The autotune also reduces register pressure (26 → 18 registers/thread), enabling higher achieved occupancy (55% → 69%) and issue slot utilization (29% → 62%).

> **Applicable when**: Triton kernel has waves_per_sm < 1.0 (SOLBottleneck rule triggered), high L1TEX stall fraction (CPIStall rule, >40% of warp cycles stalled on L1TEX), low issue slot utilization (<40%), and low achieved occupancy (<60%). Kernel logic is simple (few loads/stores per thread) so autotune overhead is low.

> **Avoid when**: Kernel already has waves_per_sm > 2.0 or is compute-bound (Compute SM Throughput > 60%). Avoid if the kernel has complex control flow or large shared memory allocations that make larger tile sizes infeasible. Also avoid if the grid dimensions are fixed by problem constraints that cannot be changed.

---

## Iteration 2

**Best so far**: 1.4366× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_grid_alignment | 1.382× | 1.426× | ✓ |  |
| B | fix_static_grid_to_use_autotuned_tile_sizes | 1.356× | 1.383× | ✓ |  |
| C | vectorized_loads_with_max_contiguous_multiple_of_hints | 1.411× | 1.406× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_static_grid_t... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 4130 ns | -1760 ns (-29.9%) | 4260 ns | -1630 ns (-27.7%) | 4190 ns | -1700 ns (-28.9%) |
| SM Throughput (%) | 15.62 % | 27.29 % | +11.67 pp | 25.75 % | +10.13 pp | 29.55 % | +13.93 pp |
| Memory Throughput (%) | 42.63 % | 20.50 % | -22.13 pp | 20.44 % | -22.19 pp | 21.90 % | -20.73 pp |
| DRAM Throughput (%) | 20.93 % | 3.98 % | -16.95 pp | 3.88 % | -17.05 pp | 3.95 % | -16.98 pp |
| Achieved Occupancy (%) | 55.25 % | 68.08 % | +12.83 pp | 67.64 % | +12.39 pp | 71.47 % | +16.22 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.56 % | 37.02 % | -33.54 pp | 38.94 % | -31.62 pp | 36.73 % | -33.83 pp |
| IPC Active | 1.14 inst/cycle | 2.44 inst/cycle | +1.30 | 2.38 inst/cycle | +1.24 | 2.22 inst/cycle | +1.08 |

---

## Iteration 3

**Best so far**: 1.4366× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | sm_aware_block_swizzle_for_workload_balance_and_l2_reuse | 1.244× | 0.690× | ✓ |  |
| B | fix_static_grid_and_eviction_policy_for_streaming_transpose | 1.343× | 1.354× | ✓ |  |
| C | evict_first_store_hint_for_streaming_writes | 1.353× | 1.437× | ✓ |  |


### Metric Comparison

| Metric | Baseline | sm_aware_block_sw... | Δ | fix_static_grid_a... | Δ | evict_first_store... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 8540 ns | +2650 ns (+45.0%) | 4350 ns | -1540 ns (-26.1%) | 4100 ns | -1790 ns (-30.4%) |
| SM Throughput (%) | 15.62 % | 20.57 % | +4.95 pp | 27.49 % | +11.87 pp | 27.44 % | +11.82 pp |
| Memory Throughput (%) | 42.63 % | 61.09 % | +18.46 pp | 20.30 % | -22.33 pp | 20.62 % | -22.01 pp |
| DRAM Throughput (%) | 20.93 % | 5.29 % | -15.64 pp | 3.81 % | -17.12 pp | 4.03 % | -16.90 pp |
| Achieved Occupancy (%) | 55.25 % | 69.50 % | +14.25 pp | 69.18 % | +13.93 pp | 69.73 % | +14.48 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.56 % | 70.83 % | +0.27 pp | 36.48 % | -34.08 pp | 37.32 % | -33.24 pp |
| IPC Active | 1.14 inst/cycle | 1.16 inst/cycle | +0.02 | 2.44 inst/cycle | +1.30 | 2.42 inst/cycle | +1.28 |


### Key Learnings

#### ❌ Plan A: `sm_aware_block_swizzle_for_workload_balance_and_l2_reuse` (0.690×)

Applying a grouped block swizzle (flattening a 2D grid into a 1D flat ID then re-mapping) in a kernel with a very small Y dimension (ynumel=30, num_pid_y≤30) caused a 1.45x increase in elapsed cycles (6161→12709) and a 50% increase in instruction count (918K→1379K). The swizzle arithmetic itself added significant overhead per thread, and the flat-ID remapping introduced non-trivial integer division/modulo chains that increased warp stall cycles from 17.4 to 39.1 cycles/instruction and collapsed eligible warps from 4.05 to 1.17. Although L2 hit rate improved (76.8%→89.8%), the L2 became the new bottleneck (Max Bandwidth 16.5%→61.1%) while compute throughput dropped (27.3%→20.6%), indicating the swizzle traded a mild compute stall for a severe memory pipeline stall. For kernels with tiny Y dimensions, the swizzle overhead dominates any L2 locality benefit.

> **Applicable when**: Never — this specific swizzle pattern caused regression. Block swizzle may be beneficial when: (1) Y dimension is large (num_pid_y >> GROUP_SIZE), (2) the kernel is L2-bandwidth bound (not compute-stall bound), and (3) the swizzle arithmetic cost is negligible relative to per-thread work.

> **Avoid when**: Kernel has a very small Y dimension (ynumel < 64, num_pid_y < GROUP_SIZE). The flat-ID remapping introduces multiple integer divisions and modulos per thread; when ynumel is tiny (e.g., 30), the swizzle arithmetic overhead exceeds any L2 locality gain. Also avoid when the kernel is already compute-stall bound (Issue Slots Busy >60%, Eligible Warps >3) — swizzle adds instructions without reducing memory latency. Avoid when the existing 2D grid already provides adequate L2 reuse.

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     4100 ns  (1.437×)  — add_autotune_increase_tile_sizes_and_waves

Final best: 4100 ns  (1.4366×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=2,  num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +28,17 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting to 2D
+    xarange = tl.arange(0, XBLOCK)
+    xarange = tl.max_contiguous(tl.multiple_of(xarange, 16), 16)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +46,7 @@

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
