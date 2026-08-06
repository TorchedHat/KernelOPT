# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 284670 ns — **Best latency**: 261820 ns (**1.0873×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_autotune_grid_and_expand_tile_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_055/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 284670 ns
- **SM Throughput**: 19.97 % | **Memory**: 79.16 % | **DRAM**: 72.48 %
- **Achieved Occupancy**: 94.01 %
- **IPC Active**: 0.80 inst/cycle | **No Eligible**: 79.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_grid_dimensions | 2.111× | 1.103× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 284670 ns | 258180 ns | -26490 ns (-9.3%) |
| SM Throughput (%) | 19.97 % | 53.93 % | +33.96 pp |
| Memory Throughput (%) | 79.16 % | 93.17 % | +14.01 pp |
| DRAM Throughput (%) | 72.48 % | 40.88 % | -31.60 pp |
| Achieved Occupancy (%) | 94.01 % | 69.92 % | -24.09 pp |
| Registers/Thread | 25.00 register/thread | 40.00 register/thread | +15 |
| Waves/SM | 59.88 | 79.84 | +19.96 |
| No Eligible (%) | 79.75 % | 45.01 % | -34.74 pp |
| IPC Active | 0.80 inst/cycle | 2.20 inst/cycle | +1.40 |

---

## Iteration 2

**Best so far**: 1.1026× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_num_stages_and_xblock_alignment_for_mio_stall_reduction | 1.563× | 1.096× | ✓ |  |
| C | evict_first_cache_policy_for_streaming_load_and_store | 2.077× | 1.085× | ✓ |  |
| D | fix_autotune_grid_and_expand_tile_configs | 2.947× | 1.087× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_num_stag... | Δ | evict_first_cache... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 284670 ns | 259840 ns | -24830 ns (-8.7%) | 262240 ns | -22430 ns (-7.9%) | 261820 ns | -22850 ns (-8.0%) |
| SM Throughput (%) | 19.97 % | 54.22 % | +34.25 pp | 55.30 % | +35.33 pp | 54.24 % | +34.27 pp |
| Memory Throughput (%) | 79.16 % | 93.67 % | +14.51 pp | 93.87 % | +14.71 pp | 93.73 % | +14.57 pp |
| DRAM Throughput (%) | 72.48 % | 40.62 % | -31.86 pp | 40.06 % | -32.42 pp | 40.33 % | -32.15 pp |
| Achieved Occupancy (%) | 94.01 % | 69.90 % | -24.11 pp | 69.88 % | -24.13 pp | 69.90 % | -24.11 pp |
| Registers/Thread | 25.00 register/thread | 40.00 register/thread | +15 | 40.00 register/thread | +15 | 40.00 register/thread | +15 |
| Waves/SM | 59.88 | 79.84 | +19.96 | 79.84 | +19.96 | 79.84 | +19.96 |
| No Eligible (%) | 79.75 % | 45.02 % | -34.73 pp | 43.99 % | -35.76 pp | 45.02 % | -34.73 pp |
| IPC Active | 0.80 inst/cycle | 2.20 inst/cycle | +1.40 | 2.24 inst/cycle | +1.44 | 2.20 inst/cycle | +1.40 |


### Key Learnings

#### ✅ Plan C: `evict_first_cache_policy_for_streaming_load_and_store` (1.085×)

For streaming kernels with near-zero L1 hit rate (≤0.05%), changing tl.load eviction_policy from 'evict_last' to 'evict_first' and adding eviction_policy='evict_first' to tl.store signals to the hardware that data has no temporal reuse, freeing L1 capacity for other warps and reducing MIO queue pressure. In this case, the NCU-reported MIO stall fraction dropped slightly (38.1% → 37.2%) and warp cycles per issued instruction improved marginally (20.26 → 19.89), yielding a modest 8.5% speedup. However, the L1/TEX throughput remained near-saturated (94.95% → 95.06%), indicating the kernel remains L1-bound and the eviction hint alone cannot fully relieve the bottleneck when the access pattern itself is still partially uncoalesced.

> **Applicable when**: L1/TEX Hit Rate ≤ 0.1% (confirming zero temporal reuse), L1/TEX Cache Throughput > 90% (L1 is saturated), CPIStall rule triggered with MIO queue stalls > 30% of stall cycles, kernel is a streaming copy/transpose with write-once output data, eviction_policy on load is currently 'evict_last'

> **Avoid when**: L1/TEX Hit Rate > 5% (data has meaningful temporal reuse and evict_last is beneficial); kernel is compute-bound rather than memory-bound; L1/TEX throughput is below 70% (L1 is not the bottleneck); the primary bottleneck is DRAM bandwidth rather than L1 saturation

---

## Iteration 3

**Best so far**: 1.1026× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_store_hints_with_correct_grid_lambda | 1.571× | 1.105× | ✓ |  |
| B | vectorized_stores_with_correct_grid_lambda_and_contiguous_hints | 2.943× | 1.088× | ✓ |  |
| C | vectorized_store_hints_with_lambda_grid | 2.077× | 1.088× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_store_... | Δ | vectorized_stores... | Δ | vectorized_store_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 284670 ns | 257600 ns | -27070 ns (-9.5%) | 261660 ns | -23010 ns (-8.1%) | 261630 ns | -23040 ns (-8.1%) |
| SM Throughput (%) | 19.97 % | 54.27 % | +34.30 pp | 54.21 % | +34.24 pp | 55.31 % | +35.34 pp |
| Memory Throughput (%) | 79.16 % | 93.79 % | +14.63 pp | 93.68 % | +14.52 pp | 93.87 % | +14.71 pp |
| DRAM Throughput (%) | 72.48 % | 41.02 % | -31.46 pp | 40.36 % | -32.12 pp | 40.16 % | -32.32 pp |
| Achieved Occupancy (%) | 94.01 % | 69.89 % | -24.12 pp | 69.90 % | -24.11 pp | 69.87 % | -24.14 pp |
| Registers/Thread | 25.00 register/thread | 40.00 register/thread | +15 | 40.00 register/thread | +15 | 40.00 register/thread | +15 |
| Waves/SM | 59.88 | 79.84 | +19.96 | 79.84 | +19.96 | 79.84 | +19.96 |
| No Eligible (%) | 79.75 % | 45.01 % | -34.74 pp | 45.03 % | -34.72 pp | 44.00 % | -35.75 pp |
| IPC Active | 0.80 inst/cycle | 2.20 inst/cycle | +1.40 | 2.20 inst/cycle | +1.40 | 2.24 inst/cycle | +1.44 |


### Key Learnings

#### ✅ Plan A: `vectorized_store_hints_with_correct_grid_lambda` (1.105×)

When a Triton kernel's store address has a contiguous dimension (stride-1 in x), applying `tl.max_contiguous(tl.multiple_of(xindex, [1, XBLOCK]), [1, XBLOCK])` to the raw 2D xindex tensor (shape [1, XBLOCK]) before decomposition enables the compiler to emit 128-bit vectorized STG instructions, reducing store transaction count. The key is to apply the hint to the raw xindex (shape [1, XBLOCK]) using per-dimension list arguments, not to a scalar or post-decomposition variable — the latter causes a shape mismatch error. The profiling metrics show a modest improvement (Duration 259.84 → 257.6 us, Elapsed Cycles 388762 → 386308) with all other metrics nearly unchanged, indicating the bottleneck remains L1/TEX saturation from uncoalesced loads rather than store throughput.

> **Applicable when**: Triton kernel has a stride-1 store dimension (MemoryCacheAccessPattern rule triggered, store bytes utilized < 32/32 per sector), xindex is a 2D tensor of shape [1, XBLOCK], and the store address uses xindex directly without post-decomposition arithmetic; L1/TEX Cache Throughput > 90% with MIO stalls > 30%

> **Avoid when**: The hint is applied to a scalar or post-decomposition variable (causes shape mismatch error); the store dimension is not contiguous in memory (stride != 1); the kernel is compute-bound rather than memory-bound; the primary bottleneck is uncoalesced loads (not stores) — in that case this hint yields only marginal gains (~1%) and the root cause requires a shared-memory transpose or grid dimension swap

#### ✅ Plan C: `vectorized_store_hints_with_lambda_grid` (1.088×)

When a Triton kernel's store address has a stride-1 dimension (x2) but NCU reports only ~28.4/32 bytes utilized per sector on global stores (MemoryCacheAccessPattern rule), applying tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK) to the 1D arange slice before reshaping — then adding the large-stride y-term separately — signals to the Triton compiler that the x-indices are contiguous and aligned, enabling 128-bit vectorized STG instructions. Critically, the hint must be applied only to the arange slice (1D), not to the full compound index expression, to avoid shape mismatch errors. In this case, the per-invocation NCU metrics changed only marginally (Warp Cycles/Instruction 20.25→19.89, Issue Slots Busy 55.0%→56.0%), suggesting the autotuner selected a slightly different config rather than the vectorization hint itself driving the gain; the kernel remains L1/TEX-bound at ~95% throughput.

> **Applicable when**: MemoryCacheAccessPattern rule triggered with store bytes utilized per sector < 30/32; kernel has a stride-1 store dimension (x-axis) with XBLOCK >= 16 (power-of-2 aligned); L1/TEX throughput > 90%; the store index is a compound expression with a large-stride y-term added to a contiguous x-term

> **Avoid when**: The hint is applied to the full compound index expression (x2 + stride*y3) rather than only the arange slice — this causes a shape mismatch compilation error; kernel is compute-bound; the store dimension is not stride-1 in memory; XBLOCK is not a power of 2 (alignment requirement for 128-bit STG)

---

## Iteration 4

**Best so far**: 1.1026× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_transpose_tile_for_coalesced_loads | 2.073× | 1.431× | ✓ |  |
| B | shared_memory_transpose_for_coalesced_load_and_store | 2.094× | 1.084× | ✓ |  |
| D | add_triton_autotune | 1.960× | 1.104× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ | shared_memory_tra... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 284670 ns | 198880 ns | -85790 ns (-30.1%) | 262660 ns | -22010 ns (-7.7%) | 257890 ns | -26780 ns (-9.4%) |
| SM Throughput (%) | 19.97 % | 37.88 % | +17.91 pp | 53.35 % | +33.38 pp | 54.23 % | +34.26 pp |
| Memory Throughput (%) | 79.16 % | 53.45 % | -25.71 pp | 90.89 % | +11.73 pp | 93.70 % | +14.54 pp |
| DRAM Throughput (%) | 72.48 % | 53.45 % | -19.03 pp | 40.00 % | -32.48 pp | 40.96 % | -31.52 pp |
| Achieved Occupancy (%) | 94.01 % | 91.92 % | -2.09 pp | 58.07 % | -35.94 pp | 69.93 % | -24.08 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 48.00 register/thread | +23 | 40.00 register/thread | +15 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 95.81 | +35.93 | 79.84 | +19.96 |
| No Eligible (%) | 79.75 % | 61.61 % | -18.14 pp | 45.98 % | -33.77 pp | 44.98 % | -34.77 pp |
| IPC Active | 0.80 inst/cycle | 1.45 inst/cycle | +0.65 | 2.16 inst/cycle | +1.36 | 2.20 inst/cycle | +1.40 |

---

## Cumulative Progress

```
Baseline:  284670 ns  (1.000×)
Iter 1     258180 ns  (1.103×)  — memory_coalescing_swap_grid_dimensions

Final best: 261820 ns  (1.0873×)
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
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 32}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
-    yoffset = tl.program_id(1) * YBLOCK
+    yoffset = tl.program_id(0) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
+    ymask = yindex < ynumel
+    xoffset = tl.program_id(1) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
+    # Apply vectorization hint to raw xindex (shape [1, XBLOCK]) for the store
+    # For 2D tensor [1, XBLOCK], pass per-dimension values as lists
+    x2_vec = tl.max_contiguous(tl.multiple_of(xindex, [1, XBLOCK]), [1, XBLOCK])
     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask, eviction_policy='evict_last')
+    # Use x2_vec (hinted) in store; x2 (unhinted) still used in load
+    tl.store(out_ptr0 + (x2_vec + 31500*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +46,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
