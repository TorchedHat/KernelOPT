# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_nhwc_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4610000 ns — **Best latency**: 24150000 ns (**0.1909×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_nhwc_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_067/synthesis/kernel_0_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32258,1,1), Waves/SM: 122.19
- **Registers/thread**: 255
- **Duration**: 4610000 ns
- **SM Throughput**: 47.82 % | **Memory**: 35.63 % | **DRAM**: 10.41 %
- **Achieved Occupancy**: 12.43 %
- **IPC Active**: 1.75 inst/cycle | **No Eligible**: 56.25 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.509× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tile_autotune_configs | 0.819× | 0.368× | ✓ |  |
| D | reduce_register_pressure_via_k_loop_reorder_and_maxnreg | 0.964× | 0.958× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4610000 ns | 9050000 ns | +4440000 ns (+96.3%) | 12520000 ns | +7910000 ns (+171.6%) | 4810000 ns | +200000 ns (+4.3%) |
| SM Throughput (%) | 47.82 % | 54.93 % | +7.11 pp | 76.16 % | +28.34 pp | 66.06 % | +18.24 pp |
| Memory Throughput (%) | 35.63 % | 23.08 % | -12.55 pp | 31.17 % | -4.46 pp | 55.61 % | +19.98 pp |
| DRAM Throughput (%) | 10.41 % | 5.30 % | -5.11 pp | 4.28 % | -6.13 pp | 13.44 % | +3.03 pp |
| Achieved Occupancy (%) | 12.43 % | 12.47 % | +0.04 pp | 24.82 % | +12.39 pp | 24.83 % | +12.40 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 128.00 register/thread | -127 | 128.00 register/thread | -127 |
| Waves/SM | 122.19 | 244.38 | +122.19 | 488.76 | +366.57 | 122.19 | +0.00 |
| No Eligible (%) | 56.25 % | 47.73 % | -8.52 pp | 28.79 % | -27.46 pp | 34.84 % | -21.41 pp |
| IPC Active | 1.75 inst/cycle | 2.09 inst/cycle | +0.34 | 2.85 inst/cycle | +1.10 | 2.61 inst/cycle | +0.86 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_k_loop_to_eliminate_uncoalesced_index_decode | 0.834× | 0.310× | ✓ |  |
| B | reorder_k_dimension_for_coalesced_memory_access | 1.148× | 0.466× | ✓ | **✓ best** |
| C | hoist_k_index_decode_out_of_loop | 0.818× | 0.368× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_k_loo... | Δ | reorder_k_dimensi... | Δ | hoist_k_index_dec... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4610000 ns | 14890000 ns | +10280000 ns (+223.0%) | 9900000 ns | +5290000 ns (+114.8%) | 12520000 ns | +7910000 ns (+171.6%) |
| SM Throughput (%) | 47.82 % | 22.09 % | -25.73 pp | 24.64 % | -23.18 pp | 76.05 % | +28.23 pp |
| Memory Throughput (%) | 35.63 % | 57.51 % | +21.88 pp | 32.62 % | -3.01 pp | 31.12 % | -4.51 pp |
| DRAM Throughput (%) | 10.41 % | 51.12 % | +40.71 pp | 22.77 % | +12.36 pp | 4.27 % | -6.14 pp |
| Achieved Occupancy (%) | 12.43 % | 24.77 % | +12.34 pp | 12.48 % | +0.05 pp | 24.82 % | +12.39 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 122.19 | 122.19 | +0.00 | 244.38 | +122.19 | 488.76 | +366.57 |
| No Eligible (%) | 56.25 % | 81.67 % | +25.42 pp | 75.19 % | +18.94 pp | 28.87 % | -27.38 pp |
| IPC Active | 1.75 inst/cycle | 0.73 inst/cycle | -1.02 | 0.99 inst/cycle | -0.76 | 2.84 inst/cycle | +1.09 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_maxnreg_cap_to_autotune_configs_to_restore_occupancy | 1.328× | 0.191× | ✓ | **✓ best** |
| C | eliminate_static_range_unroll_spill_traffic | 0.863× | 0.607× | ✓ |  |
| D | add_triton_autotune | 0.512× | 0.220× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_maxnreg_cap_t... | Δ | eliminate_static_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4610000 ns | 24150000 ns | +19540000 ns (+423.9%) | 7600000 ns | +2990000 ns (+64.9%) | 20940000 ns | +16330000 ns (+354.2%) |
| SM Throughput (%) | 47.82 % | 16.94 % | -30.88 pp | 37.54 % | -10.28 pp | 16.94 % | -30.88 pp |
| Memory Throughput (%) | 35.63 % | 60.58 % | +24.95 pp | 57.18 % | +21.55 pp | 58.33 % | +22.70 pp |
| DRAM Throughput (%) | 10.41 % | 60.58 % | +50.17 pp | 30.93 % | +20.52 pp | 58.33 % | +47.92 pp |
| Achieved Occupancy (%) | 12.43 % | 24.92 % | +12.49 pp | 24.87 % | +12.44 pp | 24.84 % | +12.41 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 128.00 register/thread | -127 | 128.00 register/thread | -127 |
| Waves/SM | 122.19 | 122.19 | +0.00 | 122.19 | +0.00 | 122.19 | +0.00 |
| No Eligible (%) | 56.25 % | 87.67 % | +31.42 pp | 62.36 % | +6.11 pp | 86.48 % | +30.23 pp |
| IPC Active | 1.75 inst/cycle | 0.49 inst/cycle | -1.26 | 1.51 inst/cycle | -0.24 | 0.54 inst/cycle | -1.21 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pre_transpose_weight_to_khkwcico_for_coalesced_weight_load | 0.863× | 0.615× | ✓ |  |
| B | reduce_num_stages_to_eliminate_pipeline_register_spill | 1.149× | 0.459× | ✓ |  |
| C | revert_nested_loops_to_flat_k_loop_implicit_gemm | 1.292× | 0.862× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pre_transpose_wei... | Δ | reduce_num_stages... | Δ | revert_nested_loo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4610000 ns | 7500000 ns | +2890000 ns (+62.7%) | 10050000 ns | +5440000 ns (+118.0%) | 5350000 ns | +740000 ns (+16.1%) |
| SM Throughput (%) | 47.82 % | 37.96 % | -9.86 pp | 24.33 % | -23.49 pp | 46.20 % | -1.62 pp |
| Memory Throughput (%) | 35.63 % | 58.31 % | +22.68 pp | 32.22 % | -3.41 pp | 26.08 % | -9.55 pp |
| DRAM Throughput (%) | 10.41 % | 31.18 % | +20.77 pp | 22.92 % | +12.51 pp | 10.02 % | -0.39 pp |
| Achieved Occupancy (%) | 12.43 % | 24.88 % | +12.45 pp | 12.48 % | +0.05 pp | 12.40 % | -0.03 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 122.19 | 122.19 | +0.00 | 244.38 | +122.19 | 244.38 | +122.19 |
| No Eligible (%) | 56.25 % | 61.98 % | +5.73 pp | 75.49 % | +19.24 pp | 55.78 % | -0.47 pp |
| IPC Active | 1.75 inst/cycle | 1.52 inst/cycle | -0.23 | 0.98 inst/cycle | -0.77 | 1.77 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan A: `pre_transpose_weight_to_khkwcico_for_coalesced_weight_load` (0.615×)

Pre-transposing the weight tensor from [OUT_C, IN_C, KH, KW] to [KH, KW, IN_C, OUT_C] before the kernel launch (and updating stride parameters accordingly) produced essentially no change in any profiling metric: duration moved from 7.6ms to 7.5ms (within noise), excessive sectors remained at 268M (60% of total), L1/TEX hit rate was unchanged at ~25%, and Warp Cycles Per Issued Instruction barely moved (10.55 → 10.45). The uncoalesced global access rule still fired with identical severity, indicating the weight layout change did not affect the actual memory access pattern. The likely cause is that the flat K-loop's weight pointer arithmetic — which uses runtime stride parameters — was already generating the same effective access pattern regardless of the physical layout, or the diff was empty (no code change was actually applied). When the dominant uncoalesced access source is not the weight tensor but rather the input activation (NHWC strided loads) or local spill traffic, transposing the weight layout cannot address the bottleneck.

> **Applicable when**: Never applicable as described — this is a negative example. Pre-transposing the weight tensor when the kernel uses a flat K-loop with runtime stride parameters does not change the observed memory access pattern or reduce excessive sectors.

> **Avoid when**: Do not apply weight pre-transposition as a fix for UncoalescedGlobalAccess when: (1) the profiling diff shows no change in excessive sectors after the layout change, (2) the flat K-loop uses runtime stride parameters (stride_wco, stride_wci) that are already passed correctly, and (3) the local load/store spill signature (1.0/32 bytes per sector) persists — indicating the dominant uncoalesced traffic is from register spills, not weight loads. In this kernel, the 268M excessive sectors and 25% L1/TEX hit rate are driven by NHWC input activation loads and spill traffic, not weight layout. Transposing the weight tensor cannot fix these sources.

#### ❌ Plan B: `reduce_num_stages_to_eliminate_pipeline_register_spill` (0.457×)

Replacing autotune configs (num_stages=2/3) with all num_stages=1 configs while simultaneously introducing larger BLOCK_M=256 options caused the autotuner to select a 256-thread block config, pushing Block Limit Registers from 2 to 1 and Block Limit Shared Mem from 3 to 1, worsening no-eligible-warp cycles from 56% to 76% and nearly doubling kernel duration (5.35ms → 10.08ms). Registers/thread remained at 255 in both cases, confirming that reducing num_stages did not reduce register pressure — the dominant effect was the autotuner selecting a larger-block config that saturated both register and shared memory limits simultaneously. The structural anti-pattern is: when a kernel is already register-limited at 255 regs/thread, adding larger BLOCK_M configs to the autotune search space without a register cap will cause the autotuner to select the larger config, further collapsing Block Limit Registers and Block Limit Shared Mem, worsening occupancy even if num_stages is reduced.

> **Applicable when**: Never applicable as described — this is a negative example. Reducing num_stages to 1 while adding larger BLOCK_M configs causes the autotuner to select a register-heavy, large-block config that worsens occupancy.

> **Avoid when**: Do not set num_stages=1 and simultaneously introduce larger BLOCK_M configs (e.g., BLOCK_M=256, num_warps=8) when the kernel is already at 255 registers/thread. The autotuner will select the larger config, pushing Block Limit Registers and Block Limit Shared Mem both to 1, halving the already-low occupancy. Additionally, do not reintroduce tl.static_range for outer kh/kw loops when the kernel is register-limited — this was the same anti-pattern that caused regressions in iterations 2, 3, and 4. The local load/store spill signature (1.0/32 bytes per sector) and 255 regs/thread are reliable indicators that the kernel is in a register-saturated regime where adding larger tiles will only worsen the situation.

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_1_configs_to_restore_flat_k_loop_occupancy | 1.149× | 1.284× | ✓ |  |
| C | small_tile_autotune_with_dynamic_range_outer_loops | 1.020× | 0.342× | ✓ |  |
| D | add_triton_autotune | 1.147× | 0.457× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_1_... | Δ | small_tile_autotu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4610000 ns | 3590000 ns | -1020000 ns (-22.1%) | 13500000 ns | +8890000 ns (+192.8%) | 10080000 ns | +5470000 ns (+118.7%) |
| SM Throughput (%) | 47.82 % | 36.88 % | -10.94 pp | 63.40 % | +15.58 pp | 24.31 % | -23.51 pp |
| Memory Throughput (%) | 35.63 % | 45.14 % | +9.51 pp | 28.09 % | -7.54 pp | 32.20 % | -3.43 pp |
| DRAM Throughput (%) | 10.41 % | 15.12 % | +4.71 pp | 3.96 % | -6.45 pp | 22.89 % | +12.48 pp |
| Achieved Occupancy (%) | 12.43 % | 12.37 % | -0.06 pp | 12.36 % | -0.07 pp | 12.48 % | +0.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 228.00 register/thread | -27 | 255.00 register/thread | +0 |
| Waves/SM | 122.19 | 244.38 | +122.19 | 977.52 | +855.33 | 244.38 | +122.19 |
| No Eligible (%) | 56.25 % | 63.05 % | +6.80 pp | 46.92 % | -9.33 pp | 75.52 % | +19.27 pp |
| IPC Active | 1.75 inst/cycle | 1.48 inst/cycle | -0.27 | 2.12 inst/cycle | +0.37 | 0.98 inst/cycle | -0.77 |


### Key Learnings

#### ❌ Plan C: `small_tile_autotune_with_dynamic_range_outer_loops` (0.342×)

Switching from large-tile configs (BLOCK_M=128-256) to exclusively small-tile configs (BLOCK_M<=64) combined with replacing tl.static_range with range() for outer kh/kw loops reduced registers/thread from 255 to 228 but failed to improve occupancy (still 12.5%, Block Limit Registers still 2). The dominant regression cause was a 4x increase in grid size (64516 → 258064 blocks) driven by smaller BLOCK_M tiles, which multiplied total executed instructions by 3.1x (1.68B → 5.16B), increasing kernel duration from 5.35ms to 13.5ms (0.34x speedup). Although L1/TEX hit rate improved dramatically (23% → 86%) due to better data reuse within smaller tiles, and warp stall cycles improved slightly (4.49 → 3.72 cycles/instruction), the sheer volume of additional work from the 4x larger grid overwhelmed all other gains. The structural anti-pattern is: reducing BLOCK_M to recover occupancy when the kernel is already register-limited by factors other than tile size causes grid explosion that multiplies total work without proportionally improving per-block efficiency.

> **Applicable when**: Never applicable as described — this is a negative example. Reducing BLOCK_M to small tiles (<=64) when the kernel is already register-limited (Block Limit Registers=2) causes grid explosion without recovering occupancy.

> **Avoid when**: Do not reduce BLOCK_M to small tiles (<=64) as an occupancy recovery strategy when Block Limit Registers is already >=2 and the register count is near 255. If the kernel's theoretical occupancy is already capped by register pressure at 12.5% and Block Limit Registers=2, reducing tile size will not improve occupancy (the register count per thread remains high at 228) but will multiply grid size by 4x, increasing total instruction count by 3x and causing severe duration regression. The signal that this strategy will fail: after applying small tiles, if registers/thread remains above 200 and Block Limit Registers remains at 2, the grid explosion dominates. Additionally, do not use this strategy when the nested kh/kw/in_c loop structure is retained — the combination of small tiles + nested loops produces 5.16B instructions vs 1.68B for the large-tile flat-K baseline.

---

## Cumulative Progress

```
Baseline:  4610000 ns  (1.000×)

Final best: 24150000 ns  (0.1909×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,22 +4,23 @@

 from triton.language.extra import libdevice
 
 # Conv kernel: implicit GEMM with NHWC layout
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
     ],
     key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
 )
 @triton.jit
 def conv2d_nhwc_kernel(
     X_ptr, W_ptr, Out_ptr,
-    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
-    KERNEL_H, KERNEL_W,
+    BATCH, IN_C: tl.constexpr, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
+    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
     stride_xn, stride_xh, stride_xw, stride_xc,
     stride_wco, stride_wci, stride_wkh, stride_wkw,
     stride_on, stride_oh, stride_ow, stride_oc,
@@ -28,43 +29,45 @@

 ):
     pid_m = tl.program_id(0)
     pid_n = tl.program_id(1)
-    
+
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    
+
     # Decode M -> (batch, out_h, out_w)
     out_w = offs_m % OUT_W
     tmp = offs_m // OUT_W
     out_h = tmp % OUT_H
     batch = tmp // OUT_H
-    
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-    
-    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
-    
+
+    valid_m = offs_m < M_TOTAL
+    valid_n = offs_n < OUT_C
+
+    # Compute TOTAL_K from constexpr params (all are constexpr so this is compile-time)
+    TOTAL_K: tl.constexpr = KERNEL_H * KERNEL_W * IN_C
+
+    # Flat K-loop: iterate over TOTAL_K = KERNEL_H * KERNEL_W * IN_C
+    # Decode kh, kw, in_c from offs_k using modulo/division arithmetic
+    # in_c is innermost (fastest-varying) for coalesced NHWC loads
     for k_off in range(0, TOTAL_K, BLOCK_K):
-        offs_k = k_off + tl.arange(0, BLOCK_K)
-        # Decode K -> (in_c, kh, kw)
+        offs_k = k_off + tl.arange(0, BLOCK_K)  # [BLOCK_K]
+        # Decode: in_c innermost (stride 1) for coalesced NHWC loads
         in_c = offs_k % IN_C
-        kw = (offs_k // IN_C) % KERNEL_W
-        kh = offs_k // (IN_C * KERNEL_W)
-        
-        ih = out_h[:, None] + kh[None, :]  # no padding, stride=1
-        iw = out_w[:, None] + kw[None, :]
-        
-        valid_m = offs_m < M_TOTAL
+        kw_idx = (offs_k // IN_C) % KERNEL_W
+        kh_idx = offs_k // (IN_C * KERNEL_W)
+        # Spatial positions broadcast to [BLOCK_M, BLOCK_K]
+        ih = out_h[:, None] + kh_idx[None, :]  # [BLOCK_M, BLOCK_K]
+        iw = out_w[:, None] + kw_idx[None, :]  # [BLOCK_M, BLOCK_K]
         valid_k = offs_k < TOTAL_K
         valid_x = valid_m[:, None] & valid_k[None, :] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
-        
         x_ptrs = X_ptr + batch[:, None] * stride_xn + ih * stride_xh + iw * stride_xw + in_c[None, :] * stride_xc
         x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
-        
-        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
-        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
+        valid_w = valid_k[:, None] & valid_n[None, :]   # [BLOCK_K, BLOCK_N]
+        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh_idx[:, None] * stride_wkh + kw_idx[:, None] * stride_wkw
         w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
-        
-        acc = tl.dot(x_tile, w_tile, acc=acc)
-    
+        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=True)
+
     # Store output in NCHW layout: [batch, out_c, out_h, out_w]
     valid_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
     out_ptrs = Out_ptr + batch[:, None] * stride_on + offs_n[None, :] * stride_oc + out_h[:, None] * stride_oh + out_w[:, None] * stride_ow
@@ -83,6 +86,7 @@

     key=['N_SPATIAL'],
 )
 @triton.jit
+
 def gelu_avgpool_kernel(
     Conv_ptr, Bias_ptr, Out_ptr,
     BATCH, OUT_C, N_SPATIAL,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
