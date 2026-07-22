# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6560 ns — **Best latency**: 12060 ns (**0.5439×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_33_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 20
- **Duration**: 6560 ns
- **SM Throughput**: 45.18 % | **Memory**: 14.50 % | **DRAM**: 9.44 %
- **Achieved Occupancy**: 66.69 %
- **IPC Active**: 2.65 inst/cycle | **No Eligible**: 34.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_eliminate_partial_wave | 1.018× | 0.981× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave | 1.003× | 0.991× | ✓ |  |
| C | autotune_xblock_to_eliminate_partial_wave | 1.073× | 0.544× | ✓ | **✓ best** |
| D | add_autotune_vectorized_loads_eliminate_partial_wave | 0.983× | 0.541× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_e... | Δ | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | add_autotune_vect... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 6690 ns | +130 ns (+2.0%) | 6620 ns | +60 ns (+0.9%) | 12060 ns | +5500 ns (+83.8%) | 12130 ns | +5570 ns (+84.9%) |
| SM Throughput (%) | 45.18 % | 47.58 % | +2.40 pp | 47.06 % | +1.88 pp | 71.45 % | +26.27 pp | 71.38 % | +26.20 pp |
| Memory Throughput (%) | 14.50 % | 14.32 % | -0.18 pp | 14.49 % | -0.01 pp | 28.32 % | +13.82 pp | 28.36 % | +13.86 pp |
| DRAM Throughput (%) | 9.44 % | 9.29 % | -0.15 pp | 9.38 % | -0.06 pp | 5.13 % | -4.31 pp | 5.11 % | -4.33 pp |
| Achieved Occupancy (%) | 66.69 % | 72.89 % | +6.20 pp | 72.47 % | +5.78 pp | 83.53 % | +16.84 pp | 83.58 % | +16.89 pp |
| Registers/Thread | 20.00 register/thread | 20.00 register/thread | +0 | 20.00 register/thread | +0 | 31.00 register/thread | +11 | 31.00 register/thread | +11 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 5.57 | +2.79 | 5.57 | +2.79 |
| No Eligible (%) | 34.65 % | 33.85 % | -0.80 pp | 30.47 % | -4.18 pp | 14.27 % | -20.38 pp | 14.38 % | -20.27 pp |
| IPC Active | 2.65 inst/cycle | 2.68 inst/cycle | +0.03 | 2.69 inst/cycle | +0.04 | 3.41 inst/cycle | +0.76 | 3.41 inst/cycle | +0.76 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_to_eliminate_partial_wave` (0.544×)

Increasing XBLOCK via autotune on a fused pointwise kernel that performs multiple non-trivial per-element computations (e.g., layer norm epilogue with rsqrt, scale, bias) caused executed instructions to balloon from ~2.4M to ~6.8M (2.8x increase) and registers per thread to jump from 20 to 31, more than doubling kernel duration (6.56 µs → 12.06 µs). Although achieved occupancy improved (66.7% → 83.5%) and waves-per-SM increased (2.78 → 5.57), the instruction count explosion dominated, resulting in a 0.54x regression. The autotune selected XBLOCK=256 rather than the expected XBLOCK=1024, and the grid was not updated to use meta['XBLOCK'], so the grid remained fixed at cdiv(xnumel, 128) blocks while each block processed 256 elements — doubling work per block without reducing block count.

> **Applicable when**: Never: this specific combination of fixed grid + autotuned XBLOCK is always wrong. More generally, avoid autotuning XBLOCK upward when: (1) the grid lambda is not updated to use meta['XBLOCK'], or (2) the kernel has significant per-element compute that causes instruction count to scale super-linearly with XBLOCK, or (3) register pressure increases enough to reduce theoretical occupancy.

> **Avoid when**: Do not autotune XBLOCK upward when the grid is not a lambda using meta['XBLOCK'] — this causes each block to process more elements without reducing block count, multiplying total work. Also avoid when the kernel has complex per-element compute (multiple transcendentals, many temporaries) that causes register spill or instruction explosion at larger tile sizes. The LaunchConfiguration NCU rule's speedup estimate assumes the grid shrinks proportionally with XBLOCK, which only holds if the grid lambda is correctly parameterized.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.995× | 1.009× | ✓ |  |
| B | persistent_tile_scheduling | 1.020× | 0.976× | ✓ |  |
| C | persistent_tile_scheduling | 0.977× | 0.915× | ✓ |  |
| D | persistent_tile_scheduling | 0.973× | 0.932× | ✓ |  |
| E | add_triton_autotune | 0.994× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 6500 ns | -60 ns (-0.9%) | 6720 ns | +160 ns (+2.4%) | 7170 ns | +610 ns (+9.3%) | 7040 ns | +480 ns (+7.3%) | 6590 ns | +30 ns (+0.5%) |
| SM Throughput (%) | 45.18 % | 47.93 % | +2.75 pp | 46.81 % | +1.63 pp | 50.31 % | +5.13 pp | 49.02 % | +3.84 pp | 47.57 % | +2.39 pp |
| Memory Throughput (%) | 14.50 % | 14.71 % | +0.21 pp | 14.35 % | -0.15 pp | 13.81 % | -0.69 pp | 13.73 % | -0.77 pp | 14.58 % | +0.08 pp |
| DRAM Throughput (%) | 9.44 % | 9.54 % | +0.10 pp | 9.28 % | -0.16 pp | 8.66 % | -0.78 pp | 8.84 % | -0.60 pp | 9.45 % | +0.01 pp |
| Achieved Occupancy (%) | 66.69 % | 72.76 % | +6.07 pp | 73.20 % | +6.51 pp | 73.31 % | +6.62 pp | 72.80 % | +6.11 pp | 71.43 % | +4.74 pp |
| Registers/Thread | 20.00 register/thread | 30.00 register/thread | +10 | 30.00 register/thread | +10 | 30.00 register/thread | +10 | 30.00 register/thread | +10 | 20.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 34.65 % | 31.06 % | -3.59 pp | 29.49 % | -5.16 pp | 29.54 % | -5.11 pp | 28.16 % | -6.49 pp | 30.18 % | -4.47 pp |
| IPC Active | 2.65 inst/cycle | 2.74 inst/cycle | +0.09 | 2.76 inst/cycle | +0.11 | 2.77 inst/cycle | +0.12 | 2.75 inst/cycle | +0.10 | 2.68 inst/cycle | +0.03 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_multiply_add_to_fma_instructions | 1.018× | 1.005× | ✓ |  |
| B | fuse_fp32_instructions_via_reciprocal_multiply | 1.016× | 0.981× | ✓ |  |
| C | streaming_eviction_policy_for_large_strided_load | 0.959× | 0.932× | ✓ |  |
| D | fuse_fp32_multiply_add_to_fma_instructions | 0.976× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_multipl... | Δ | fuse_fp32_instruc... | Δ | streaming_evictio... | Δ | fuse_fp32_multipl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6560 ns | 6530 ns | -30 ns (-0.5%) | 6690 ns | +130 ns (+2.0%) | 7040 ns | +480 ns (+7.3%) | 6690 ns | +130 ns (+2.0%) |
| SM Throughput (%) | 45.18 % | 48.64 % | +3.46 pp | 47.64 % | +2.46 pp | 50.70 % | +5.52 pp | 50.61 % | +5.43 pp |
| Memory Throughput (%) | 14.50 % | 14.69 % | +0.19 pp | 14.32 % | -0.18 pp | 13.65 % | -0.85 pp | 14.36 % | -0.14 pp |
| DRAM Throughput (%) | 9.44 % | 9.51 % | +0.07 pp | 9.27 % | -0.17 pp | 8.79 % | -0.65 pp | 9.29 % | -0.15 pp |
| Achieved Occupancy (%) | 66.69 % | 73.09 % | +6.40 pp | 72.97 % | +6.28 pp | 74.59 % | +7.90 pp | 72.92 % | +6.23 pp |
| Registers/Thread | 20.00 register/thread | 30.00 register/thread | +10 | 30.00 register/thread | +10 | 30.00 register/thread | +10 | 30.00 register/thread | +10 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 34.65 % | 29.39 % | -5.26 pp | 33.02 % | -1.63 pp | 26.72 % | -7.93 pp | 27.88 % | -6.77 pp |
| IPC Active | 2.65 inst/cycle | 2.73 inst/cycle | +0.08 | 2.74 inst/cycle | +0.09 | 2.84 inst/cycle | +0.19 | 2.75 inst/cycle | +0.10 |

---

## Cumulative Progress

```
Baseline:  6560 ns  (1.000×)

Final best: 12060 ns  (0.5439×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,33 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 752640
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x0 = (xindex % 32)
-    x1 = ((xindex // 32) % 588)
-    x2 = xindex // 18816
-    x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 2)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 2)), xmask)
-    tmp1 = tl.load(in_ptr1 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
-    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 384.0, tl.float32)
-    tmp5 = (tmp3 / tmp4)
-    tmp6 = tl.full([1], 1e-05, tl.float32)
-    tmp7 = tmp5 + tmp6
-    tmp8 = libdevice.rsqrt(tmp7)
-    tmp9 = tmp2 * tmp8
-    tmp11 = tmp9 * tmp10
-    tmp13 = tmp11 + tmp12
-    tl.store(out_ptr0 + (x3), tmp13, xmask)
-
-
+    pid = tl.program_id(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    for tile_id in tl.range(pid, num_tiles, tl.num_programs(0)):
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x0 = (xindex % 32)
+        x1 = ((xindex // 32) % 588)
+        x2 = xindex // 18816
+        x3 = xindex
+        tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 2)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 2)), xmask)
+        tmp1 = tl.load(in_ptr1 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
+        tmp3 = tl.load(in_ptr2 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
+        tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
+        tmp2 = tmp0 - tmp1
+        tmp4 = tl.full([1], 384.0, tl.float32)
+        tmp5 = (tmp3 / tmp4)
+        tmp6 = tl.full([1], 1e-05, tl.float32)
+        tmp7 = tmp5 + tmp6
+        tmp8 = libdevice.rsqrt(tmp7)
+        tmp9 = tmp2 * tmp8
+        tmp11 = tmp9 * tmp10
+        tmp13 = tmp11 + tmp12
+        tl.store(out_ptr0 + (x3), tmp13, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
