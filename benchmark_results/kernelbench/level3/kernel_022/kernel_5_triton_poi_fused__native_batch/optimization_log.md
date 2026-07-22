# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 67010 ns — **Best latency**: 65980 ns (**1.0156×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_cache_reuse_and_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 32
- **Duration**: 67010 ns
- **SM Throughput**: 25.77 % | **Memory**: 61.94 % | **DRAM**: 55.57 %
- **Achieved Occupancy**: 92.46 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | software_pipelining_via_autotune | 1.828× | 1.028× | ✓ |  |
| C | autotune_tile_sizes_for_cache_reuse_and_parallelism | 2.011× | 1.016× | ✓ | **✓ best** |
| D | autotune_tile_sizes_and_pipeline_stages | 1.486× | 0.876× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 67010 ns | 65150 ns | -1860 ns (-2.8%) | 65980 ns | -1030 ns (-1.5%) | 76450 ns | +9440 ns (+14.1%) |
| SM Throughput (%) | 25.77 % | 26.16 % | +0.39 pp | 26.15 % | +0.38 pp | 40.12 % | +14.35 pp |
| Memory Throughput (%) | 61.94 % | 63.45 % | +1.51 pp | 63.19 % | +1.25 pp | 86.70 % | +24.76 pp |
| DRAM Throughput (%) | 55.57 % | 57.94 % | +2.37 pp | 56.72 % | +1.15 pp | 19.96 % | -35.61 pp |
| Achieved Occupancy (%) | 92.46 % | 92.73 % | +0.27 pp | 92.38 % | -0.08 pp | 47.11 % | -45.35 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 58.00 register/thread | +26 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 22.27 | +11.13 |
| No Eligible (%) | 72.49 % | 72.63 % | +0.14 pp | 72.73 % | +0.24 pp | 57.45 % | -15.04 pp |
| IPC Active | 1.08 inst/cycle | 1.09 inst/cycle | +0.01 | 1.08 inst/cycle | +0.00 | 1.70 inst/cycle | +0.62 |

---

## Iteration 2

**Best so far**: 1.0156× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_channel_as_innermost_warp_dimension | 1.199× | 0.286× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 67010 ns | 234140 ns | +167130 ns (+249.4%) |
| SM Throughput (%) | 25.77 % | 42.01 % | +16.24 pp |
| Memory Throughput (%) | 61.94 % | 71.64 % | +9.70 pp |
| DRAM Throughput (%) | 55.57 % | 14.49 % | -41.08 pp |
| Achieved Occupancy (%) | 92.46 % | 11.76 % | -80.70 pp |
| Registers/Thread | 32.00 register/thread | 238.00 register/thread | +206 |
| Waves/SM | 11.14 | 89.09 | +77.95 |
| No Eligible (%) | 72.49 % | 57.29 % | -15.20 pp |
| IPC Active | 1.08 inst/cycle | 1.71 inst/cycle | +0.63 |


### Key Learnings

#### ❌ Plan C: `memory_coalescing_channel_as_innermost_warp_dimension` (0.286×)

Transposing the XBLOCK/YBLOCK tile assignment to make the stride-1 dimension innermost (for memory coalescing) caused registers per thread to explode from 32 to 238, collapsing theoretical occupancy from 100% to 12.5% and achieved occupancy from 92.7% to 11.8%. The 3.6x increase in elapsed cycles (97K → 351K) and 5.7x increase in executed instructions (13.5M → 77.9M) indicate the register pressure from holding both [YBLOCK,1] and [1,XBLOCK] shaped tensors simultaneously in a 2D tiled layout overwhelmed any coalescing benefit. When a kernel already has high occupancy (>90%) and moderate register usage (~32/thread), restructuring the 2D tile layout to transpose which axis is innermost can cause the compiler to spill or allocate far more registers to handle the new broadcast shapes, making the cure worse than the disease.

> **Applicable when**: Never: this specific axis-swap approach caused a 3.5x regression due to register explosion. Do not apply when the kernel already achieves >90% occupancy with low register count (~32/thread) and the 2D tile layout would require holding large broadcast tensors simultaneously.

> **Avoid when**: Any kernel where the 2D tile axis swap would increase registers per thread beyond ~64. Specifically: when XBLOCK and YBLOCK are both large (e.g., 64+), the [YBLOCK,XBLOCK] broadcast tile requires O(XBLOCK*YBLOCK) register slots, causing occupancy collapse. The coalescing gain (~1-2x) cannot compensate for an 8x occupancy drop. Also avoid when the kernel already has high occupancy (>80%) — the baseline is already well-utilized and restructuring risks register pressure.

---

## Iteration 3

**Best so far**: 1.0156× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | revert_2d_tile_axis_to_restore_occupancy | 1.122× | 0.343× | ✓ |  |
| E | add_triton_autotune | 0.702× | 0.622× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_2d_tile_ax... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 67010 ns | 195650 ns | +128640 ns (+192.0%) | 107780 ns | +40770 ns (+60.8%) |
| SM Throughput (%) | 25.77 % | 50.48 % | +24.71 pp | 44.44 % | +18.67 pp |
| Memory Throughput (%) | 61.94 % | 84.07 % | +22.13 pp | 84.94 % | +23.00 pp |
| DRAM Throughput (%) | 55.57 % | 17.35 % | -38.22 pp | 31.54 % | -24.03 pp |
| Achieved Occupancy (%) | 92.46 % | 17.60 % | -74.86 pp | 23.11 % | -69.35 pp |
| Registers/Thread | 32.00 register/thread | 161.00 register/thread | +129 | 126.00 register/thread | +94 |
| Waves/SM | 11.14 | 59.39 | +48.25 | 44.55 | +33.41 |
| No Eligible (%) | 72.49 % | 48.87 % | -23.62 pp | 54.51 % | -17.98 pp |
| IPC Active | 1.08 inst/cycle | 2.06 inst/cycle | +0.98 | 1.82 inst/cycle | +0.74 |


### Key Learnings

#### ❌ Plan C: `revert_2d_tile_axis_to_restore_occupancy` (0.343×)

Attempting to revert a prior axis-swap regression by swapping XBLOCK/YBLOCK roles back produced only a partial improvement: registers dropped from 238 to 161 (not the expected ~32), and theoretical occupancy rose from 12.5% to 18.75% (not the expected ~100%). The kernel remained 3x slower than the true baseline (97K cycles), settling at 291K cycles. The revert did not reproduce the original low-register state because the surrounding code context (e.g., shared memory allocation of 32KB/block, compiler decisions) had changed, preventing the compiler from recovering the original register allocation. This confirms that once a 2D tile axis-swap regression is introduced, a simple axis-swap revert is insufficient to restore the original register count — the kernel must be fully restored to its original source, not just the tile index assignments.

> **Applicable when**: Never: this revert strategy failed to restore the original register count or occupancy. The partial axis-swap revert only moved registers from 238 to 161 and occupancy from 12.5% to 18.75%, still far from the 32-register, 92.7%-occupancy baseline.

> **Avoid when**: Any situation where the goal is to restore a prior fast baseline after a 2D tile axis-swap regression. A partial axis-swap revert (swapping only the index shape assignments) is insufficient — the full original kernel source must be restored, including any shared memory, grid launch, and compiler hint context that influenced the original register allocation.

#### ❌ Plan E: `add_triton_autotune` (0.622×)

Reducing the autotune config space (from 9 configs to 4, and dropping 'ynumel' from the autotune key) caused registers per thread to drop from 238 to 126 and theoretical occupancy to double from 12.5% to 25%, and elapsed cycles dropped from 351K to 161K. However, the overall wall-clock speedup was only 0.62x relative to the true baseline (before the 2D axis-swap regression), because the kernel body still uses the register-expensive 2D [YBLOCK,XBLOCK] broadcast tile layout. Autotuning config parameters alone cannot fix a structural register-pressure problem caused by the 2D tile layout; the kernel body itself must be restructured (e.g., flattened to 1D) to achieve low register counts and high occupancy.

> **Applicable when**: Never as a standalone fix: reducing autotune configs or changing the autotune key cannot resolve register pressure caused by a 2D broadcast tile layout. The structural kernel body must change.

> **Avoid when**: When the kernel's register pressure stems from the 2D [YBLOCK,XBLOCK] broadcast tile layout (both dimensions held simultaneously in registers). Autotuning config parameters (XBLOCK, YBLOCK, num_warps, num_stages) can shift the register count somewhat but cannot eliminate the O(XBLOCK*YBLOCK) register allocation inherent to the 2D tile. Do not apply autotune-only fixes when the root cause is tile layout, not parameter values.

---

## Cumulative Progress

```
Baseline:  67010 ns  (1.000×)
Iter 1     65980 ns  (1.016×)  — autotune_tile_sizes_for_cache_reuse_and_parallelism

Final best: 65980 ns  (1.0156×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,38 +5,56 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 256}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1920
     xnumel = 12544
+    # XBLOCK covers channel+batch dimension (ynumel=1920, stride-1 in in_ptr0)
+    # Consecutive warp threads differ in channel (y0, stride-1) -> coalesced
+    xoffset = tl.program_id(0) * XBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape [1, XBLOCK]
+    xmask = xindex < ynumel
+    # YBLOCK covers spatial dimension (xnumel=12544, stride-192 in in_ptr0)
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    # Apply scalar hints to 1D arange before broadcasting to 2D
-    xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
-    xindex = xrange[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 192)
-    y1 = yindex // 192
-    y3 = yindex
-    # Hoist channel-only scalar loads BEFORE the main element load (ymask only, no xmask needed)
-    tmp_mean    = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
-    tmp_var     = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
-    tmp_weight  = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
-    tmp_bias    = tl.load(in_ptr4 + (y0), mask=ymask, eviction_policy='evict_last')   # [YBLOCK, 1]
-    # Precompute per-channel scale and bias (amortized over XBLOCK elements)
+    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape [YBLOCK, 1]
+    ymask = yindex < xnumel
+    # Index decomposition
+    y0 = xindex % 192       # channel index [0, 192), shape [1, XBLOCK]
+    y1 = xindex // 192      # batch index [0, 10), shape [1, XBLOCK]
+    x2 = yindex             # spatial index [0, 12544), shape [YBLOCK, 1]
+    y3 = xindex             # channel+batch combined index, shape [1, XBLOCK]
+    # Channel parameter loads: y0 varies across XBLOCK (warp threads) -> coalesced
+    tmp_mean   = tl.load(in_ptr1 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
+    tmp_var    = tl.load(in_ptr2 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
+    tmp_weight = tl.load(in_ptr3 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
+    tmp_bias   = tl.load(in_ptr4 + y0, mask=xmask, eviction_policy='evict_last')   # [1, XBLOCK]
+    # Precompute per-channel scale and bias
     tmp_eps     = tl.full([1, 1], 1e-05, tl.float32)
     tmp_inv_std = libdevice.rsqrt(tmp_var + tmp_eps)        # rsqrt(var + eps)
-    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)  [YBLOCK, 1]
-    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale        [YBLOCK, 1]
-    # Load main input element
-    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 2408448*y1), xmask & ymask, eviction_policy='evict_first')
-    # Single FMA: x * combined_scale + combined_bias  (broadcasts [YBLOCK,1] over XBLOCK)
+    combined_scale = tmp_weight * tmp_inv_std               # weight / sqrt(var+eps)  [1, XBLOCK]
+    combined_bias  = tmp_bias - tmp_mean * combined_scale   # bias - mean*scale        [1, XBLOCK]
+    # Load main input element: address formula y0 + 192*x2 + 2408448*y1 UNCHANGED
+    # Now consecutive warp threads differ in y0 (stride-1) -> coalesced
+    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 2408448*y1), mask=xmask & ymask, eviction_policy='evict_first')
+    # FMA: broadcasts [1, XBLOCK] over [YBLOCK, 1]
     tmp15 = tmp0 * combined_scale + combined_bias
-    tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, xmask & ymask)
-
+    # Store: address formula x2 + 12544*y3 UNCHANGED
+    tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -48,7 +66,7 @@

     ynumel = 1920
     xnumel = 12544
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
