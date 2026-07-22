# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3710 ns — **Best latency**: 3870 ns (**0.9587×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_30_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1715,1,1), Waves/SM: 0.81
- **Registers/thread**: 18
- **Duration**: 3710 ns
- **SM Throughput**: 12.03 % | **Memory**: 7.94 % | **DRAM**: 4.88 %
- **Achieved Occupancy**: 55.51 %
- **IPC Active**: 1.37 inst/cycle | **No Eligible**: 64.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_wave_fragmentation | 1.023× | 1.104× | ✓ |  |
| B | autotune_xblock_num_warps_num_stages | 1.064× | 1.054× | ✓ | **✓ best** |
| C | increase_xblock_tile_size_with_autotune | 1.013× | 1.114× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 1.022× | 0.739× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3360 ns | -350 ns (-9.4%) | 3520 ns | -190 ns (-5.1%) | 3330 ns | -380 ns (-10.2%) | 5020 ns | +1310 ns (+35.3%) |
| SM Throughput (%) | 12.03 % | 18.06 % | +6.03 pp | 13.25 % | +1.22 pp | 18.38 % | +6.35 pp | 44.38 % | +32.35 pp |
| Memory Throughput (%) | 7.94 % | 12.23 % | +4.29 pp | 8.36 % | +0.42 pp | 12.35 % | +4.41 pp | 20.55 % | +12.61 pp |
| DRAM Throughput (%) | 4.88 % | 5.45 % | +0.57 pp | 5.14 % | +0.26 pp | 5.50 % | +0.62 pp | 3.59 % | -1.29 pp |
| Achieved Occupancy (%) | 55.51 % | 59.17 % | +3.66 pp | 60.77 % | +5.26 pp | 60.48 % | +4.97 pp | 55.52 % | +0.01 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 34.00 register/thread | +16 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 2.17 | +1.36 |
| No Eligible (%) | 64.76 % | 42.08 % | -22.68 pp | 62.11 % | -2.65 pp | 41.52 % | -23.24 pp | 29.20 % | -35.56 pp |
| IPC Active | 1.37 inst/cycle | 2.04 inst/cycle | +0.67 | 1.35 inst/cycle | -0.02 | 1.92 inst/cycle | +0.55 | 2.78 inst/cycle | +1.41 |

---

## Iteration 2

**Best so far**: 1.054× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.030× | 1.008× | ✓ |  |
| B | persistent_tile_scheduling_for_underutilized_pointwise_kernel | 0.947× | 1.016× | ✓ |  |
| C | persistent_tile_scheduling | 1.012× | 1.016× | ✓ |  |
| D | persistent_tile_scheduling | 1.019× | 0.942× | ✓ |  |
| E | add_triton_autotune | 0.993× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3680 ns | -30 ns (-0.8%) | 3650 ns | -60 ns (-1.6%) | 3650 ns | -60 ns (-1.6%) | 3940 ns | +230 ns (+6.2%) | 3680 ns | -30 ns (-0.8%) |
| SM Throughput (%) | 12.03 % | 12.12 % | +0.09 pp | 16.32 % | +4.29 pp | 16.30 % | +4.27 pp | 14.81 % | +2.78 pp | 12.17 % | +0.14 pp |
| Memory Throughput (%) | 7.94 % | 7.96 % | +0.02 pp | 8.34 % | +0.40 pp | 8.29 % | +0.35 pp | 8.20 % | +0.26 pp | 8.10 % | +0.16 pp |
| DRAM Throughput (%) | 4.88 % | 4.93 % | +0.05 pp | 4.99 % | +0.11 pp | 5.00 % | +0.12 pp | 4.61 % | -0.27 pp | 4.96 % | +0.08 pp |
| Achieved Occupancy (%) | 55.51 % | 56.11 % | +0.60 pp | 56.56 % | +1.05 pp | 56.70 % | +1.19 pp | 59.04 % | +3.53 pp | 59.03 % | +3.52 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 64.76 % | 65.08 % | +0.32 pp | 61.61 % | -3.15 pp | 58.96 % | -5.80 pp | 63.13 % | -1.63 pp | 65.28 % | +0.52 pp |
| IPC Active | 1.37 inst/cycle | 1.37 inst/cycle | +0.00 | 1.51 inst/cycle | +0.14 | 1.50 inst/cycle | +0.13 | 1.43 inst/cycle | +0.06 | 1.32 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.054× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_multiply_add_to_fma_in_batchnorm_pointwise | 1.017× | 1.025× | ✓ |  |
| B | explicit_fma_promotion_to_reduce_non_fused_fp32_instructions | 0.969× | 1.036× | ✓ |  |
| C | fuse_fp32_multiply_add_to_fma_instruction | 1.031× | 1.016× | ✓ |  |
| D | replace_libdevice_rsqrt_with_native_tl_rsqrt_for_fma_fusion | 1.030× | 1.045× | ✓ |  |
| E | add_triton_autotune | 1.073× | 0.959× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_fp32_multipl... | Δ | explicit_fma_prom... | Δ | fuse_fp32_multipl... | Δ | replace_libdevice... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3620 ns | -90 ns (-2.4%) | 3580 ns | -130 ns (-3.5%) | 3650 ns | -60 ns (-1.6%) | 3550 ns | -160 ns (-4.3%) | 3870 ns | +160 ns (+4.3%) |
| SM Throughput (%) | 12.03 % | 16.19 % | +4.16 pp | 15.62 % | +3.59 pp | 12.75 % | +0.72 pp | 14.92 % | +2.89 pp | 16.25 % | +4.22 pp |
| Memory Throughput (%) | 7.94 % | 8.28 % | +0.34 pp | 7.97 % | +0.03 pp | 8.45 % | +0.51 pp | 8.27 % | +0.33 pp | 8.19 % | +0.25 pp |
| DRAM Throughput (%) | 4.88 % | 5.05 % | +0.17 pp | 5.07 % | +0.19 pp | 4.96 % | +0.08 pp | 5.09 % | +0.21 pp | 4.67 % | -0.21 pp |
| Achieved Occupancy (%) | 55.51 % | 57.99 % | +2.48 pp | 62.15 % | +6.64 pp | 55.83 % | +0.32 pp | 55.79 % | +0.28 pp | 58.66 % | +3.15 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 30.00 register/thread | +12 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 64.76 % | 63.26 % | -1.50 pp | 62.57 % | -2.19 pp | 64.10 % | -0.66 pp | 61.89 % | -2.87 pp | 58.77 % | -5.99 pp |
| IPC Active | 1.37 inst/cycle | 1.55 inst/cycle | +0.18 | 1.55 inst/cycle | +0.18 | 1.35 inst/cycle | -0.02 | 1.41 inst/cycle | +0.04 | 1.55 inst/cycle | +0.18 |


### Key Learnings

#### ✅ Plan D: `replace_libdevice_rsqrt_with_native_tl_rsqrt_for_fma_fusion` (1.045×)

When the FPInstructions NCU rule flags a high ratio of non-fused to fused FP32 instructions (e.g., 3:1), replacing `libdevice.rsqrt` with `tl.math.rsqrt` maps the call to a single PTX `rsqrt.approx.f32` hardware instruction instead of a multi-instruction Newton-Raphson software path, reducing total executed instructions by ~11% (432K → 384K) and kernel duration by ~2.7% (3.65 → 3.55 µs). The libdevice path expands to multiple FMUL/FFMA instructions that block the LLVM backend from fusing surrounding multiply-add operations into FMAs; switching to `tl.math.rsqrt` removes this barrier and allows the compiler to emit FMA chains for adjacent arithmetic. This optimization is safe for inference-only batch normalization where ~2 ULP relative error from the approximate hardware rsqrt is acceptable.

> **Applicable when**: FPInstructions NCU rule is triggered with a high non-fused to fused FP32 instruction ratio (>2:1); kernel contains `libdevice.rsqrt` (or other libdevice transcendentals) in a hot arithmetic chain; the operation is used in inference (no_training) batch norm or similar where approximate rsqrt (~2 ULP) is numerically acceptable.

> **Avoid when**: Full IEEE-754 precision rsqrt is required (e.g., training with gradient accumulation where small errors compound); the libdevice call is not the dominant source of non-fused instructions; kernel is already using tl.math.rsqrt or hardware intrinsics; the surrounding arithmetic has no fusable multiply-add pairs that would benefit from FMA emission.

---

## Iteration 4

**Best so far**: 1.054× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | force_fma_fusion_via_explicit_tl_math_fma_calls | 1.031× | 1.045× | ✓ |  |
| B | autotune_xblock_within_existing_persistent_kernel_loop | 1.038× | 1.159× | ✓ |  |
| C | activate_persistent_kernel_by_capping_grid_to_num_sms | 1.023× | 1.036× | ✓ |  |
| D | evict_first_on_streaming_main_tensor_load_to_reduce_l1tex_stalls | 1.001× | 1.054× | ✓ |  |


### Metric Comparison

| Metric | Baseline | force_fma_fusion_... | Δ | autotune_xblock_w... | Δ | activate_persiste... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3550 ns | -160 ns (-4.3%) | 3200 ns | -510 ns (-13.7%) | 3580 ns | -130 ns (-3.5%) | 3520 ns | -190 ns (-5.1%) |
| SM Throughput (%) | 12.03 % | 14.69 % | +2.66 pp | 13.03 % | +1.00 pp | 16.12 % | +4.09 pp | 12.76 % | +0.73 pp |
| Memory Throughput (%) | 7.94 % | 7.99 % | +0.05 pp | 8.80 % | +0.86 pp | 8.18 % | +0.24 pp | 7.81 % | -0.13 pp |
| DRAM Throughput (%) | 4.88 % | 5.11 % | +0.23 pp | 5.68 % | +0.80 pp | 5.05 % | +0.17 pp | 5.15 % | +0.27 pp |
| Achieved Occupancy (%) | 55.51 % | 55.76 % | +0.25 pp | 45.22 % | -10.29 pp | 55.98 % | +0.47 pp | 60.05 % | +4.54 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 32.00 register/thread | +14 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 64.76 % | 62.34 % | -2.42 pp | 67.94 % | +3.18 pp | 62.26 % | -2.50 pp | 62.21 % | -2.55 pp |
| IPC Active | 1.37 inst/cycle | 1.38 inst/cycle | +0.01 | 1.14 inst/cycle | -0.23 | 1.48 inst/cycle | +0.11 | 1.51 inst/cycle | +0.14 |

---

## Iteration 5

**Best so far**: 1.054× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_streaming_main_tensor_to_improve_broadcast_l2_hit_rate | 1.039× | 1.104× | ✓ |  |
| B | eliminate_nonpow2_modulo_via_2d_grid_decomposition | 0.199× | 0.152× | ✓ |  |
| C | remove_unnecessary_persistent_loop_to_reduce_register_pressure | 1.034× | 1.025× | ✓ |  |
| D | flatten_persistent_loop_and_evict_first_streaming_load | 1.029× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | eliminate_nonpow2... | Δ | remove_unnecessar... | Δ | flatten_persisten... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3360 ns | -350 ns (-9.4%) | 24320 ns | +20610 ns (+555.5%) | 3620 ns | -90 ns (-2.4%) | 3710 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 12.03 % | 12.95 % | +0.92 pp | 2.34 % | -9.69 pp | 13.37 % | +1.34 pp | 14.03 % | +2.00 pp |
| Memory Throughput (%) | 7.94 % | 8.79 % | +0.85 pp | 19.45 % | +11.51 pp | 8.43 % | +0.49 pp | 8.35 % | +0.41 pp |
| DRAM Throughput (%) | 4.88 % | 5.42 % | +0.54 pp | 0.07 % | -4.81 pp | 5.01 % | +0.13 pp | 4.89 % | +0.01 pp |
| Achieved Occupancy (%) | 55.51 % | 46.89 % | -8.62 pp | 50.42 % | -5.09 pp | 54.69 % | -0.82 pp | 59.54 % | +4.03 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 64.76 % | 64.50 % | -0.26 pp | 97.88 % | +33.12 pp | 63.50 % | -1.26 pp | 67.02 % | +2.26 pp |
| IPC Active | 1.37 inst/cycle | 1.19 inst/cycle | -0.18 | 0.08 inst/cycle | -1.29 | 1.29 inst/cycle | -0.08 | 1.49 inst/cycle | +0.12 |


### Key Learnings

#### ❌ Plan B: `eliminate_nonpow2_modulo_via_2d_grid_decomposition` (0.152×)

Replacing a 1D flat grid (with x0 = xindex % 112) with a 2D grid (pid0=channel, pid1=spatial tile) to eliminate the non-power-of-2 modulo caused a 6.6x regression. The 2D decomposition changed the innermost memory access pattern from contiguous (stride-1 along xindex) to strided (stride-112 along spatial_index * 112 + x0), triggering the UncoalescedGlobalAccess rule with 82% excessive sectors (384K of 466K sectors wasted). Warp Cycles Per Issued Instruction exploded from 25.9 to 465.7 cycles, and the CPIStall rule shifted from L1TEX dependency stalls to EXIT-drain stalls (284.6 cycles, 61.1% of stall budget), indicating warps were blocked waiting for uncoalesced store operations to drain at kernel exit. The original 1D layout had coalesced access (contiguous xindex maps to contiguous memory addresses given the tensor's channel-stride-1 layout), while the 2D layout broke coalescing by making threads within a warp access addresses separated by 112 floats (448 bytes).

> **Applicable when**: Never: this 2D grid decomposition pattern is harmful when the tensor's innermost (stride-1) dimension is the channel dimension and threads within a warp are assigned different spatial indices, causing inter-thread stride of channel_count * sizeof(float) bytes.

> **Avoid when**: Always avoid 2D grid decomposition that assigns pid0=channel and pid1=spatial when the tensor layout has channel as the innermost stride (NHWC-like with C innermost). This creates stride-C uncoalesced access. Only use 2D grids when the decomposition preserves coalescing: e.g., pid0=batch/spatial (outermost) and pid1=channel-tile (innermost), so threads within a warp access consecutive channel indices.

---

## Cumulative Progress

```
Baseline:  3710 ns  (1.000×)
Iter 1     3520 ns  (1.054×)  — autotune_xblock_num_warps_num_stages

Final best: 3870 ns  (0.9587×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,44 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 219520
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = xindex < xnumel
-    x2 = xindex
-    x0 = (xindex % 112)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp5 = tmp3 + 1e-05
-    tmp10 = libdevice.rsqrt(tmp5)
-    tmp11 = tmp2 * tmp10
-    tmp15 = tmp11 * tmp12 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
+    pid = tl.program_id(0)
+    num_cta = tl.num_programs(0)
+    num_tiles = tl.cdiv(xnumel, XBLOCK)
+    tile_id = pid
+    while tile_id < num_tiles:
+        xoffset = tile_id * XBLOCK
+        xindex = xoffset + tl.arange(0, XBLOCK)[:]
+        xmask = xindex < xnumel
+        x2 = xindex
+        x0 = (xindex % 112)
+        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+        tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+        tmp2 = tmp0 - tmp1
+        tmp5 = tmp3 + 1e-05
+        tmp10 = libdevice.rsqrt(tmp5)
+        tmp11 = tmp2 * tmp10
+        tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
+        tl.store(in_out_ptr0 + (x2), tmp15, xmask)
+        tile_id += num_cta
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -34,7 +52,7 @@

     arg_4 = inputs[4]
     xnumel = 219520
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
