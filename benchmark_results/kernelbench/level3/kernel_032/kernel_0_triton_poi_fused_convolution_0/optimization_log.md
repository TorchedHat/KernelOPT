# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 3550 ns (**1.6676×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_032/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.63 % | **Memory**: 41.09 % | **DRAM**: 20.87 %
- **Achieved Occupancy**: 55.17 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.364× | 1.778× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.454× | 1.668× | ✓ | **✓ best** |
| C | memory_coalescing_swap_tile_dimensions | 1.400× | 1.313× | ✓ |  |
| D | increase_tile_sizes_and_autotune_for_coalescing_and_utilization | 1.310× | 1.330× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 3330 ns | -2590 ns (-43.8%) | 3550 ns | -2370 ns (-40.0%) | 4510 ns | -1410 ns (-23.8%) | 4450 ns | -1470 ns (-24.8%) |
| SM Throughput (%) | 16.63 % | 16.45 % | -0.18 pp | 17.72 % | +1.09 pp | 25.52 % | +8.89 pp | 21.26 % | +4.63 pp |
| Memory Throughput (%) | 41.09 % | 20.71 % | -20.38 pp | 19.23 % | -21.86 pp | 22.43 % | -18.66 pp | 31.47 % | -9.62 pp |
| DRAM Throughput (%) | 20.87 % | 5.00 % | -15.87 pp | 4.64 % | -16.23 pp | 6.83 % | -14.04 pp | 14.77 % | -6.10 pp |
| Achieved Occupancy (%) | 55.17 % | 43.71 % | -11.46 pp | 44.21 % | -10.96 pp | 52.96 % | -2.21 pp | 45.44 % | -9.73 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 28.00 register/thread | +2 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 54.91 % | -15.48 pp | 54.47 % | -15.92 pp | 38.94 % | -31.45 pp | 56.53 % | -13.86 pp |
| IPC Active | 1.13 inst/cycle | 1.66 inst/cycle | +0.53 | 1.70 inst/cycle | +0.57 | 2.38 inst/cycle | +1.25 | 1.67 inst/cycle | +0.54 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_autotune` (1.778×)

For layout permutation kernels (e.g. NCHW→NHWC) where loads are stride-1 contiguous but stores are non-coalesced, applying tl.max_contiguous + tl.multiple_of hints on the load index enables the compiler to emit 128-bit vectorized loads, cutting L1TEX scoreboard stall cycles roughly in half (29.8→15.6 cycles/issued instruction) and reducing total instruction count by ~44%. Adding @triton.autotune over XBLOCK/YBLOCK configs simultaneously allows the runtime to select the tile shape that best amortizes the fixed overhead of non-coalesced stores, yielding a 1.78x end-to-end speedup. This strategy is most effective when CPIStall reports ≥40% of cycles stalled on L1TEX scoreboard dependencies and UncoalescedGlobalAccess reports ≥15% excessive sectors on the store side.

> **Applicable when**: CPIStall reports ≥40% of warp cycles stalled on L1TEX scoreboard dependencies; UncoalescedGlobalAccess reports ≥15% excessive sectors; load access pattern is stride-1 contiguous in the innermost thread dimension; kernel is a data layout permutation (e.g. NCHW→NHWC); waves_per_sm < 1.0 indicating underutilization

> **Avoid when**: Load pattern is not stride-1 contiguous (hints would be incorrect and could cause wrong results); kernel already uses autotune; xnumel is a power of 2 and boundary masking can be removed (different optimization path); store pattern is also coalesced (bottleneck is elsewhere)

---

## Iteration 2

**Best so far**: 1.6676× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_broadcast_axes_for_coalesced_stores | 1.263× | 1.352× | ✓ |  |
| B | vectorized_loads | 1.427× | 1.778× | ✓ |  |
| C | vectorized_loads_contiguity_hints | 1.300× | 1.293× | ✓ |  |
| D | vectorized_loads_contiguity_hints | 1.397× | 1.352× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_broadcast_ax... | Δ | vectorized_loads | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 4380 ns | -1540 ns (-26.0%) | 3330 ns | -2590 ns (-43.8%) | 4580 ns | -1340 ns (-22.6%) | 4380 ns | -1540 ns (-26.0%) |
| SM Throughput (%) | 16.63 % | 25.88 % | +9.25 pp | 17.51 % | +0.88 pp | 21.25 % | +4.62 pp | 25.77 % | +9.14 pp |
| Memory Throughput (%) | 41.09 % | 22.71 % | -18.38 pp | 20.52 % | -20.57 pp | 30.52 % | -10.57 pp | 22.66 % | -18.43 pp |
| DRAM Throughput (%) | 20.87 % | 7.06 % | -13.81 pp | 5.00 % | -15.87 pp | 14.43 % | -6.44 pp | 7.06 % | -13.81 pp |
| Achieved Occupancy (%) | 55.17 % | 53.03 % | -2.14 pp | 43.58 % | -11.59 pp | 47.37 % | -7.80 pp | 52.83 % | -2.34 pp |
| Registers/Thread | 26.00 register/thread | 28.00 register/thread | +2 | 18.00 register/thread | -8 | 27.00 register/thread | +1 | 28.00 register/thread | +2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 40.81 % | -29.58 pp | 59.04 % | -11.35 pp | 59.05 % | -11.34 pp | 39.25 % | -31.14 pp |
| IPC Active | 1.13 inst/cycle | 2.37 inst/cycle | +1.24 | 1.67 inst/cycle | +0.54 | 1.70 inst/cycle | +0.57 | 2.35 inst/cycle | +1.22 |


### Key Learnings

#### ✅ Plan A: `swap_broadcast_axes_for_coalesced_stores` (1.361×)

When a layout-permutation kernel has uncoalesced stores (UncoalescedGlobalAccess: 50% excess sectors, MemoryCacheAccessPattern: 10.7/32 bytes per sector) because the innermost thread axis maps to a non-unit output stride, swapping the 2D tile broadcast axes — making the stride-1 output dimension innermost ([None, :]) and the high-stride dimension outermost ([:, None]) — coalesces stores and raises L1/TEX hit rate from 0% to ~44%. This also improves Issue Slots Busy from ~44% to ~60% and reduces No-Eligible warp cycles from ~59% to ~39%, yielding a ~1.36x speedup despite higher register count (18→28) and more instructions due to more work per thread. The strategy is effective even when the load side becomes strided after the swap, provided the load dimension is small (e.g., ynumel≤32) and L2 hit rate is already high (>70%), limiting load latency increase.

> **Applicable when**: UncoalescedGlobalAccess fires with >=50% excess sectors; MemoryCacheAccessPattern shows <16 bytes utilized per 32-byte sector for global stores; the output tensor has stride-1 along the Y dimension but Y is mapped to the outermost tile axis while X (with non-unit output stride) is the innermost thread axis; the strided load dimension after the swap is small (e.g., ynumel<=32) and L2 hit rate is already high (>70%)

> **Avoid when**: Stores are already coalesced (bytes-per-sector near 32); swapping axes would make a large load dimension strided with low L2 hit rate, causing load latency to dominate; the kernel is compute-bound rather than memory-latency-bound; YBLOCK values required after the swap are not powers of 2

#### ✅ Plan C: `vectorized_loads_contiguity_hints` (1.293×)

Applying tl.max_contiguous + tl.multiple_of hints on the load index without also changing autotune configs caused a regression (1.29x slowdown relative to the prior state): elapsed cycles increased from 4896 to 6725, executed instructions nearly doubled (429K→730K), and registers per thread jumped from 18 to 27, reducing scheduling efficiency. The contiguity hints appear to have interacted poorly with the new autotune config set (which also changed in this diff), increasing register pressure and instruction count rather than enabling vectorized loads. When autotune configs are simultaneously changed alongside contiguity hints, the combined effect can be negative if the new configs select larger tiles that increase register spill or instruction overhead beyond the savings from vectorization.

> **Applicable when**: Never: this combination (contiguity hints + simultaneous autotune config change) caused regression; the hints alone may be valid but cannot be isolated from the config change in this diff

> **Avoid when**: When autotune is already present and performing well; when adding contiguity hints simultaneously with autotune config changes, as the interaction can increase register pressure (18→27 registers/thread) and instruction count (429K→730K), negating any vectorization benefit

#### ✅ Plan D: `vectorized_loads_contiguity_hints` (1.293×)

Applying tl.max_contiguous + tl.multiple_of hints on the load index without also changing autotune configs caused a regression (1.29x slowdown relative to the prior state): elapsed cycles increased from 4896 to 6725, executed instructions nearly doubled (429K→730K), and registers per thread jumped from 18 to 27, reducing scheduling efficiency. The contiguity hints appear to have interacted poorly with the new autotune config set (which also changed in this diff), increasing register pressure and instruction count rather than enabling vectorized loads. When autotune configs are simultaneously changed alongside contiguity hints, the combined effect can be negative if the new configs select larger tiles that increase register spill or instruction overhead beyond the savings from vectorization.

> **Applicable when**: Never: this combination (contiguity hints + simultaneous autotune config change) caused regression; the hints alone may be valid but cannot be isolated from the config change in this diff

> **Avoid when**: When autotune is already present and performing well; when adding contiguity hints simultaneously with autotune config changes, as the interaction can increase register pressure (18→27 registers/thread) and instruction count (429K→730K), negating any vectorization benefit

---

## Iteration 3

**Best so far**: 1.6676× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | pipeline_stages_and_streaming_eviction_for_latency_hiding | 1.159× | 1.433× | ✓ |  |
| C | coalesced_store_swap_broadcast_axes_for_stride1_output | 1.397× | 1.361× | ✓ |  |
| D | eviction_policy_and_pipeline_stages_for_streaming_transpose | 1.278× | 1.284× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pipeline_stages_a... | Δ | coalesced_store_s... | Δ | eviction_policy_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5920 ns | 4130 ns | -1790 ns (-30.2%) | 4350 ns | -1570 ns (-26.5%) | 4610 ns | -1310 ns (-22.1%) |
| SM Throughput (%) | 16.63 % | 17.18 % | +0.55 pp | 24.37 % | +7.74 pp | 21.09 % | +4.46 pp |
| Memory Throughput (%) | 41.09 % | 27.08 % | -14.01 pp | 21.43 % | -19.66 pp | 30.46 % | -10.63 pp |
| DRAM Throughput (%) | 20.87 % | 14.89 % | -5.98 pp | 7.12 % | -13.75 pp | 14.38 % | -6.49 pp |
| Achieved Occupancy (%) | 55.17 % | 55.97 % | +0.80 pp | 52.80 % | -2.37 pp | 47.62 % | -7.55 pp |
| Registers/Thread | 26.00 register/thread | 22.00 register/thread | -4 | 28.00 register/thread | +2 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.39 % | 63.38 % | -7.01 pp | 39.08 % | -31.31 pp | 57.31 % | -13.08 pp |
| IPC Active | 1.13 inst/cycle | 1.32 inst/cycle | +0.19 | 2.36 inst/cycle | +1.23 | 1.75 inst/cycle | +0.62 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)
Iter 1     3550 ns  (1.668×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 3550 ns  (1.6676×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 1}, num_warps=2),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +26,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Contiguity hints on 1D range BEFORE [None,:] reshape -> enables 128-bit vectorized loads
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

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
