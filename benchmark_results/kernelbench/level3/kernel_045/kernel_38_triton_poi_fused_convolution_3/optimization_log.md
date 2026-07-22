# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_38` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 71740 ns — **Best latency**: 71170 ns (**1.0080×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_38`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_38_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 27
- **Duration**: 71740 ns
- **SM Throughput**: 18.45 % | **Memory**: 84.62 % | **DRAM**: 70.27 %
- **Achieved Occupancy**: 91.85 %
- **IPC Active**: 0.76 inst/cycle | **No Eligible**: 81.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | — | 1.008× | ✓ | **✓ best** |
| B | autotune_tile_shape_for_transpose_coalescing | — | 2.413× | ✓ |  |
| D | autotune_tile_sizes_for_coalesced_transpose | — | 2.406× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_sha... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 71740 ns | 71170 ns | -570 ns (-0.8%) | 29730 ns | -42010 ns (-58.6%) | 29820 ns | -41920 ns (-58.4%) |
| SM Throughput (%) | 18.45 % | 18.14 % | -0.31 pp | 57.48 % | +39.03 pp | 56.24 % | +37.79 pp |
| Memory Throughput (%) | 84.62 % | 85.29 % | +0.67 pp | 50.33 % | -34.29 pp | 49.28 % | -35.34 pp |
| DRAM Throughput (%) | 70.27 % | 70.78 % | +0.51 pp | 30.86 % | -39.41 pp | 30.75 % | -39.52 pp |
| Achieved Occupancy (%) | 91.85 % | 91.81 % | -0.04 pp | 73.37 % | -18.48 pp | 73.09 % | -18.76 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 26.00 register/thread | -1 | 26.00 register/thread | -1 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 81.01 % | 81.17 % | +0.16 pp | 43.60 % | -37.41 pp | 43.61 % | -37.40 pp |
| IPC Active | 0.76 inst/cycle | 0.75 inst/cycle | -0.01 | 2.26 inst/cycle | +1.50 | 2.25 inst/cycle | +1.49 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_shape_for_transpose_coalescing` (2.413×)

When a transpose/copy kernel shows L2 Cache Throughput >90%, L1 hit rate <1%, and Compute SM Throughput <20% with CPIStall dominated by L1TEX scoreboard waits, the root cause is non-coalesced memory stores caused by a tile shape that is wide in the non-coalesced output dimension. Replacing a fixed asymmetric tile (e.g., XBLOCK=128, YBLOCK=8) with @triton.autotune over multiple square and near-square tile configurations (e.g., 32x32, 64x16, 16x64) allows the runtime to select a shape that aligns warp threads with the stride-1 output dimension, converting scattered multi-sector store transactions into coalesced 128-byte transactions. This reduced L2 throughput from 94% to 47%, warp stall cycles from 77 to 20 cycles/instruction, and improved Compute SM Throughput from 18% to 57%, yielding a 2.4x speedup.

> **Applicable when**: Triton transpose/copy kernel with: L2 Cache Throughput >80%, L1 hit rate <1%, Compute SM Throughput <25%, CPIStall rule triggered with L1TEX scoreboard stalls >40% of CPI, and a fixed asymmetric tile shape (one dimension much larger than the other) that is wide in the non-coalesced output dimension.

> **Avoid when**: Kernel is already compute-bound (Compute SM Throughput >70%); tile shape is already square or aligned with the stride-1 output dimension; problem dimensions are not exact multiples of candidate block sizes and masking overhead would dominate; kernel has data reuse patterns that benefit from the evict_last policy.

#### ✅ Plan D: `autotune_tile_sizes_for_coalesced_transpose` (2.406×)

When a transpose kernel shows L2 throughput near saturation (>90%), near-zero L1 hit rate (<1%), and >80% scheduler idle time due to L1TEX scoreboard stalls (~41 cycles/warp), the root cause is non-coalesced stores from mismatched tile shapes. Adding @triton.autotune over (XBLOCK, YBLOCK) pairs—including square tiles like (32,32) and (64,32)—allows the autotuner to find configurations where the store dimension spans contiguous output addresses per warp, reducing L2 throughput from 94% to 46% and warp stall cycles from 77 to 21. Removing the evict_last eviction policy on streaming loads also eliminates unnecessary cache pollution.

> **Applicable when**: Triton transpose or permutation kernel with: L2 Cache Throughput >85%, L1/TEX Hit Rate <1%, Scheduler 'No Eligible' >75%, Warp Cycles Per Issued Instruction >50, Compute (SM) Throughput <25%, CPIStall rule triggered on L1TEX. Fixed tile sizes (XBLOCK, YBLOCK) are hardcoded and not tuned for the output stride pattern.

> **Avoid when**: Kernel is already compute-bound (Compute SM Throughput >70%). Kernel does not perform a transpose (load and store strides are both contiguous). Problem size is too small for autotuning overhead to amortize. YBLOCK must evenly divide ynumel for all candidate configs—verify divisibility before adding configs.

---

## Iteration 2

**Best so far**: 1.008× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipelining_via_num_stages_to_hide_l1tex_latency | — | 2.408× | ✓ |  |
| B | vectorized_loads_with_multiple_of_max_contiguous_hints | — | 2.408× | ✓ |  |
| C | evict_first_streaming_with_software_pipelining | — | 0.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | vectorized_loads_... | Δ | evict_first_strea... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 71740 ns | 29790 ns | -41950 ns (-58.5%) | 29790 ns | -41950 ns (-58.5%) | 77220 ns | +5480 ns (+7.6%) |
| SM Throughput (%) | 18.45 % | 57.33 % | +38.88 pp | 57.50 % | +39.05 pp | 18.25 % | -0.20 pp |
| Memory Throughput (%) | 84.62 % | 50.15 % | -34.47 pp | 50.35 % | -34.27 pp | 79.29 % | -5.33 pp |
| DRAM Throughput (%) | 70.27 % | 30.91 % | -39.36 pp | 30.90 % | -39.37 pp | 64.58 % | -5.69 pp |
| Achieved Occupancy (%) | 91.85 % | 72.42 % | -19.43 pp | 75.22 % | -16.63 pp | 91.92 % | +0.07 pp |
| Registers/Thread | 27.00 register/thread | 26.00 register/thread | -1 | 26.00 register/thread | -1 | 28.00 register/thread | +1 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 81.01 % | 43.61 % | -37.40 pp | 43.67 % | -37.34 pp | 80.56 % | -0.45 pp |
| IPC Active | 0.76 inst/cycle | 2.22 inst/cycle | +1.46 | 2.27 inst/cycle | +1.51 | 0.77 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan A: `software_pipelining_via_num_stages_to_hide_l1tex_latency` (2.408×)

Adding num_stages=2 and num_stages=3 to @triton.autotune configs for a simple load+store transpose kernel produced no measurable improvement: Duration changed from 29.73us to 29.79us, Eligible Warps Per Scheduler moved only from 2.31 to 2.45, and No Eligible remained at ~43.6%. Triton's software pipelining via num_stages is most effective when there are sequential tile iterations with independent loads that can be prefetched ahead of computation; a single-tile-per-block load+store kernel with no inner loop has no prefetch opportunity to exploit, so num_stages has no effect on L1TEX scoreboard stall reduction. The 2.4x reported speedup is attributable to the prior tile-shape autotuning change, not to num_stages.

> **Applicable when**: Never applicable as a standalone optimization for single-tile load+store kernels with no inner loop. Document as a negative result to avoid re-attempting.

> **Avoid when**: Kernel body is a single load+store per block with no inner loop over tiles — num_stages cannot pipeline anything because there is no next iteration to prefetch. Also avoid when the kernel already has L1TEX stalls that are not reducible by prefetching (e.g., stalls from a single unavoidable memory access per thread).

#### ❌ Plan B: `vectorized_loads_with_multiple_of_max_contiguous_hints` (2.408×)

Adding tl.max_contiguous and tl.multiple_of hints to the load index of a transpose kernel produced no measurable improvement when the kernel is already bottlenecked by non-coalesced stores rather than load latency. All key metrics (No Eligible %, Warp Cycles Per Issued Instruction, L2 throughput, Duration) remained essentially unchanged after the hints were applied, indicating the load path was not the binding constraint. When the CPIStall bottleneck is dominated by store-side L1TEX scoreboard stalls (non-coalesced output writes), vectorization hints on the load side cannot reduce the stall depth.

> **Applicable when**: Never effective in isolation when the kernel is store-bound (non-coalesced writes dominate L1TEX stalls). May help in load-bound kernels where L1/TEX Hit Rate is low and loads are on the critical path.

> **Avoid when**: Kernel is already bottlenecked by non-coalesced stores (CPIStall rule triggered, No Eligible >40%, and the store address has large strides between consecutive threads). In this regime, load-side vectorization hints have no effect on the binding bottleneck and should not be applied.

---

## Iteration 3

**Best so far**: 1.008× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | contiguous_hints_for_vectorized_loads_and_stores | — | 2.408× | ✓ |  |
| B | increase_num_warps_and_num_stages_to_hide_l1tex_latency | — | 2.411× | ✓ |  |
| C | autotune_num_warps_and_eviction_policy_for_transpose_store_latency_hiding | — | 0.956× | ✓ |  |


### Metric Comparison

| Metric | Baseline | contiguous_hints_... | Δ | increase_num_warp... | Δ | autotune_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 71740 ns | 29790 ns | -41950 ns (-58.5%) | 29760 ns | -41980 ns (-58.5%) | 75040 ns | +3300 ns (+4.6%) |
| SM Throughput (%) | 18.45 % | 56.96 % | +38.51 pp | 57.89 % | +39.44 pp | 20.77 % | +2.32 pp |
| Memory Throughput (%) | 84.62 % | 49.88 % | -34.74 pp | 50.64 % | -33.98 pp | 80.97 % | -3.65 pp |
| DRAM Throughput (%) | 70.27 % | 30.84 % | -39.43 pp | 30.86 % | -39.41 pp | 66.87 % | -3.40 pp |
| Achieved Occupancy (%) | 91.85 % | 73.15 % | -18.70 pp | 72.01 % | -19.84 pp | 91.99 % | +0.14 pp |
| Registers/Thread | 27.00 register/thread | 26.00 register/thread | -1 | 26.00 register/thread | -1 | 28.00 register/thread | +1 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 81.01 % | 43.61 % | -37.40 pp | 44.21 % | -36.80 pp | 78.43 % | -2.58 pp |
| IPC Active | 0.76 inst/cycle | 2.25 inst/cycle | +1.49 | 2.23 inst/cycle | +1.47 | 0.87 inst/cycle | +0.11 |


### Key Learnings

#### ❌ Plan B: `increase_num_warps_and_num_stages_to_hide_l1tex_latency` (2.411×)

Adding num_warps=16 and num_stages=4 configs to @triton.autotune for a load+store transpose kernel produced no measurable improvement: Duration changed from 29.79us to 29.76us (~0.1%), No Eligible actually increased from 43.61% to 44.21%, and Eligible Warps Per Scheduler remained flat at ~2.4. The kernel's L1TEX scoreboard stalls are caused by non-coalesced stores (stride-128 output writes), not by insufficient warp count — adding more warps per block does not change the per-warp store access pattern or reduce the store latency. When the CPIStall bottleneck is structural (non-coalesced output stride), occupancy-based latency hiding via num_warps expansion cannot eliminate the root cause and should not be attempted.

> **Applicable when**: Never effective for single-tile load+store transpose kernels where CPIStall is caused by non-coalesced stores with large output strides. The bottleneck is structural (store address pattern), not warp-count limited.

> **Avoid when**: Kernel is a simple load+store transpose with no inner loop and non-coalesced output writes (store stride >> 1 element between consecutive threads). In this regime, increasing num_warps from 8 to 16 or num_stages from 3 to 4 has no effect because the autotuner already selects the best tile shape and the remaining stalls are irreducible without changing the store access pattern itself.

---

## Cumulative Progress

```
Baseline:  71740 ns  (1.000×)
Iter 1     71170 ns  (1.008×)  — vectorized_loads_with_contiguous_hints

Final best: 71170 ns  (1.0080×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,40 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1024
     xnumel = 32768
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
+    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 32768*y3), None, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 4194304*y1), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 32768*y3), xmask & ymask)
+    tl.store(out_ptr0 + (y0 + 128*x2 + 4194304*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +46,7 @@

     ynumel = 1024
     xnumel = 32768
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_38[grid](arg_0, arg_1, 1024, 32768, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_38[grid](arg_0, arg_1, 1024, 32768)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
