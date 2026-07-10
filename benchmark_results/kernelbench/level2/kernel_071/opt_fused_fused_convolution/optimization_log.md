# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_div_leaky_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 10980000 ns — **Best latency**: 6080000 ns (**1.8059×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_div_leaky_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_071/synthesis/kernel_0_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7938,1,1), Waves/SM: 30.07
- **Registers/thread**: 255
- **Duration**: 10980000 ns
- **SM Throughput**: 45.76 % | **Memory**: 63.12 % | **DRAM**: 1.22 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.86 inst/cycle | **No Eligible**: 53.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_loop_restructure_and_maxnreg | 2.866× | 1.797× | ✓ | **✓ best** |
| C | annotate_loads_stores_with_cache_eviction_policies | 1.001× | 1.005× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_configs_and_maxnreg | 0.614× | 0.513× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | annotate_loads_st... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10980000 ns | 6110000 ns | -4870000 ns (-44.4%) | 10930000 ns | -50000 ns (-0.5%) | 21410000 ns | +10430000 ns (+95.0%) |
| SM Throughput (%) | 45.76 % | 18.61 % | -27.15 pp | 46.08 % | +0.32 pp | 26.91 % | -18.85 pp |
| Memory Throughput (%) | 63.12 % | 51.15 % | -11.97 pp | 63.48 % | +0.36 pp | 98.14 % | +35.02 pp |
| DRAM Throughput (%) | 1.22 % | 44.88 % | +43.66 pp | 1.07 % | -0.15 pp | 0.54 % | -0.68 pp |
| Achieved Occupancy (%) | 12.46 % | 24.84 % | +12.38 pp | 12.48 % | +0.02 pp | 24.91 % | +12.45 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 128.00 register/thread | -127 |
| Waves/SM | 30.07 | 30.07 | +0.00 | 30.07 | +0.00 | 60.14 | +30.07 |
| No Eligible (%) | 53.57 % | 81.11 % | +27.54 pp | 53.28 % | -0.29 pp | 73.03 % | +19.46 pp |
| IPC Active | 1.86 inst/cycle | 0.76 inst/cycle | -1.10 | 1.87 inst/cycle | +0.01 | 1.08 inst/cycle | -0.78 |

---

## Iteration 2

**Best so far**: 1.7971× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | restructure_nested_loops_for_register_reduction_with_maxnreg | 2.847× | 1.169× | ✓ |  |
| C | restructure_fused_loop_to_nested_static_range_loops | 2.873× | 0.674× | ✓ |  |
| D | add_triton_autotune | 2.875× | 1.806× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | restructure_neste... | Δ | restructure_fused... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 10980000 ns | 9390000 ns | -1590000 ns (-14.5%) | 16290000 ns | +5310000 ns (+48.4%) | 6080000 ns | -4900000 ns (-44.6%) |
| SM Throughput (%) | 45.76 % | 16.17 % | -29.59 pp | 18.30 % | -27.46 pp | 18.78 % | -26.98 pp |
| Memory Throughput (%) | 63.12 % | 60.70 % | -2.42 pp | 66.09 % | +2.97 pp | 51.67 % | -11.45 pp |
| DRAM Throughput (%) | 1.22 % | 60.70 % | +59.48 pp | 51.82 % | +50.60 pp | 44.99 % | +43.77 pp |
| Achieved Occupancy (%) | 12.46 % | 24.86 % | +12.40 pp | 24.92 % | +12.46 pp | 24.84 % | +12.38 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 128.00 register/thread | -127 | 128.00 register/thread | -127 |
| Waves/SM | 30.07 | 15.03 | -15.04 | 60.14 | +30.07 | 30.07 | +0.00 |
| No Eligible (%) | 53.57 % | 86.89 % | +33.32 pp | 85.00 % | +31.43 pp | 80.97 % | +27.40 pp |
| IPC Active | 1.86 inst/cycle | 0.52 inst/cycle | -1.34 | 0.60 inst/cycle | -1.26 | 0.76 inst/cycle | -1.10 |


### Key Learnings

#### ❌ Plan C: `restructure_fused_loop_to_nested_static_range_loops` (0.674×)

Applying the nested tl.static_range loop restructure to a kernel that already used this pattern (from a prior iteration) caused a 1.49x regression. The diff shows the actual change was a swap of autotune tile configs (from larger BLOCK_M=128 tiles to smaller BLOCK_M=32-64 tiles) combined with a doubling of the grid size (15876→31752, waves_per_sm 30→60). This doubled the total instruction count (898M→1927M) and worsened shared memory bank conflicts from 7.9-way to 12.0-way (uncoalesced shared wavefronts 66%→75%), while eligible warps per scheduler dropped from 0.33 to 0.19. The compute throughput remained flat at ~18%, confirming the regression was driven by increased memory traffic from smaller tiles and worse shared memory access patterns, not by any benefit from the loop restructure itself.

> **Applicable when**: Never: this is a negative result. Do not re-apply the nested static_range restructure when the kernel already uses it. Do not shrink BLOCK_M below 64 when shared memory bank conflicts are already ≥7-way.

> **Avoid when**: The kernel already uses nested tl.static_range loops (the restructure was applied in a prior iteration). Shrinking tile sizes (BLOCK_M from 128→32) when shared memory bank conflicts are already high (≥7-way) worsens conflicts further (to 12-way) and doubles grid size, increasing total memory traffic without any compute benefit. The bottleneck in this state is shared memory bank conflicts and MIO stalls, not loop structure or register pressure.

---

## Iteration 3

**Best so far**: 1.8059× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | match_block_k_to_in_c_and_increase_pipeline_stages | 2.846× | 1.163× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_transposed_w_load | 2.874× | 0.679× | ✓ |  |


### Metric Comparison

| Metric | Baseline | match_block_k_to_... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10980000 ns | 9440000 ns | -1540000 ns (-14.0%) | 16180000 ns | +5200000 ns (+47.4%) |
| SM Throughput (%) | 45.76 % | 16.09 % | -29.67 pp | 18.26 % | -27.50 pp |
| Memory Throughput (%) | 63.12 % | 60.45 % | -2.67 pp | 65.94 % | +2.82 pp |
| DRAM Throughput (%) | 1.22 % | 60.45 % | +59.23 pp | 52.67 % | +51.45 pp |
| Achieved Occupancy (%) | 12.46 % | 24.85 % | +12.39 pp | 24.92 % | +12.46 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 128.00 register/thread | -127 |
| Waves/SM | 30.07 | 15.03 | -15.04 | 60.14 | +30.07 |
| No Eligible (%) | 53.57 % | 86.95 % | +33.38 pp | 84.95 % | +31.38 pp |
| IPC Active | 1.86 inst/cycle | 0.52 inst/cycle | -1.34 | 0.60 inst/cycle | -1.26 |


### Key Learnings

#### ❌ Plan C: `eliminate_shared_memory_bank_conflicts_via_transposed_w_load` (0.679×)

Loading the W tile transposed as [BLOCK_N, BLOCK_K] and applying tl.trans() before tl.dot was intended to eliminate shared memory bank conflicts, but instead worsened them from 7.9-way to 12.0-way and increased uncoalesced shared wavefronts from 66% to 75%. The actual regression was driven by the simultaneous swap to smaller autotune tile configs (BLOCK_M reduced from 128 to 32-64), which doubled the grid size (15876→31752, waves_per_sm 30→60) and total instruction count (898M→1927M), while eligible warps per scheduler dropped from 0.33 to 0.19 and warp cycles per issued instruction rose from 20.89 to 26.48. Compute throughput remained flat at ~18%, confirming the regression was caused by increased memory traffic and worse shared memory access patterns from smaller tiles, not by the transposed load itself.

> **Applicable when**: Never: this direction has now failed twice (timed out in iteration 2 as swizzle, regressed in iteration 3 as transposed load). The bank conflict root cause is real but the fix consistently causes regressions, likely because the autotune config changes accompanying the W layout change select smaller tiles that worsen the overall access pattern.

> **Avoid when**: This kernel already has 7-12 way shared memory bank conflicts and the autotune config space includes small tiles (BLOCK_M<=64). Changing W tile layout from [BLOCK_K, BLOCK_N] to [BLOCK_N, BLOCK_K] with tl.trans() does not reduce bank conflicts in practice — NCU shows conflicts worsened from 7.9-way to 12.0-way after the change. The confounding factor is that smaller tile configs (BLOCK_M=32-64) double the grid size and total instruction count, overwhelming any bank-conflict benefit. Do not attempt this direction again without first locking the autotune configs to the best large-tile config (BLOCK_M=128, BLOCK_N=64) to isolate the layout change effect.

---

## Iteration 4

**Best so far**: 1.8059× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 2.847× | 1.167× | ✓ |  |
| D | tma_loads_via_block_ptr_for_w_tile | 2.841× | 0.697× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 10980000 ns | 9410000 ns | -1570000 ns (-14.3%) | 15760000 ns | +4780000 ns (+43.5%) |
| SM Throughput (%) | 45.76 % | 16.15 % | -29.61 pp | 17.98 % | -27.78 pp |
| Memory Throughput (%) | 63.12 % | 60.63 % | -2.49 pp | 66.63 % | +3.51 pp |
| DRAM Throughput (%) | 1.22 % | 60.63 % | +59.41 pp | 50.63 % | +49.41 pp |
| Achieved Occupancy (%) | 12.46 % | 24.85 % | +12.39 pp | 24.92 % | +12.46 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 128.00 register/thread | -127 |
| Waves/SM | 30.07 | 15.03 | -15.04 | 60.14 | +30.07 |
| No Eligible (%) | 53.57 % | 86.91 % | +33.34 pp | 84.57 % | +31.00 pp |
| IPC Active | 1.86 inst/cycle | 0.52 inst/cycle | -1.34 | 0.62 inst/cycle | -1.24 |


### Key Learnings

#### ⚠️ Plan A: `tf32_tensor_cores` (0.000×)

Plan 'tf32_tensor_cores' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=9.119e-03 > atol=0.001
seed=42 output[0]: max_diff=7.724e-03 > atol=0.001
seed=1337 output[0]: max_diff=7.354e-03 > atol=0.001

> **Applicable when**: The kernel achieves only 9% of FP32 peak and 18.26% compute SOL while spending 84.95% of scheduler cycles with no eligible warp. The dominant stall is L1TEX (47.5% of warp cycles) caused by shared memory bank conflicts (12-way) and uncoalesced shared accesses (75% excessive wavefronts). Currently tl.dot uses allow_tf32=False, forcing the compiler to emit FP32 FFMA instructions that run on CUDA cores at ~1/8 the throughput of TF32 Tensor Core operations. Enabling allow_tf32=True switches tl.dot to use Tensor Cores (TF32 mode), which deliver ~8x higher throughput for matrix multiply. This dramatically reduces the number of compute cycles needed per tile, giving the memory subsystem more time to service pending loads and reducing the effective pressure on the L1TEX pipeline. The K dimension is only 72 (IN_C=8 x KH=3 x KW=3), so TF32 mantissa truncation (23->10 bits) accumulates negligible rounding error within the rtol=1e-3, atol=1e-5 correctness tolerance.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=9.119e-03 > atol=0.001
seed=42 output[0]: max_diff=7.724e-03 > atol=0.001
seed=1337 output[0]: max_diff=7.354e-03 > atol=0.001

#### ❌ Plan D: `tma_loads_via_block_ptr_for_w_tile` (0.697×)

Replacing scalar pointer arithmetic for the W tile with tl.make_block_ptr + tl.load(boundary_check) was intended to leverage TMA hardware on Hopper for async copies and hardware swizzling to eliminate 5.3-way shared memory bank conflicts. Instead, bank conflicts worsened from 5.3-way to 12.0-way and uncoalesced shared wavefronts grew from 54% to 75%, while the grid size quadrupled (7938→31752, waves_per_sm 15→60) due to the autotune config swap to smaller tiles (BLOCK_M=32-64 replacing BLOCK_M=128-256). The quadrupled grid size doubled total executed instructions (950M→1914M) and increased kernel duration from 9.44ms to 15.76ms. The confounding factor is that the autotune config set was simultaneously changed to smaller tiles alongside the block_ptr change, making it impossible to isolate the TMA effect; the regression is primarily attributable to the smaller tile configs driving a 4x larger grid.

> **Applicable when**: Never in this kernel's current state: this direction has now failed. tl.make_block_ptr for the W tile consistently co-occurs with smaller autotune tile configs that quadruple the grid size and worsen bank conflicts. The TMA hardware swizzling benefit (if any) is overwhelmed by the increased memory traffic from smaller tiles.

> **Avoid when**: This kernel already has shared memory bank conflicts (≥5-way) and the autotune config set includes small tiles (BLOCK_M≤64). Switching the W tile load from scalar pointer arithmetic to tl.make_block_ptr while simultaneously reducing tile sizes causes a 4x grid size increase, doubles total instruction count, and worsens bank conflicts from 5.3-way to 12.0-way. Do not attempt this direction again without locking autotune configs to the best large-tile config (BLOCK_M=128-256) to isolate the block_ptr effect from the tile-size effect. The pattern of confounding tile-size changes with memory access pattern changes has now caused regressions in iterations 2, 3, and 4.

---

## Cumulative Progress

```
Baseline:  10980000 ns  (1.000×)
Iter 1     6110000 ns  (1.797×)  — reduce_register_pressure_via_loop_restructure_and_maxnreg
Iter 2     6080000 ns  (1.806×)  — add_triton_autotune

Final best: 6080000 ns  (1.8059×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,14 +2,19 @@

 import triton
 import triton.language as tl
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=5, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=6, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=5, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=5, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=6, num_warps=8, maxnreg=128),
     ],
     key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
 )
@@ -28,18 +33,18 @@

     # W is in OHWI layout: (OUT_C, KH, KW, IN_C) -> but we use (OUT_C, KH*KW*IN_C) effectively
     # Actually W in channels-last: (OUT_C, IN_C, KH, KW) reordered to (OUT_C, KH, KW, IN_C)
     # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
-    
+
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
     out_h = (offs_m // OUT_W) % OUT_H
     batch = offs_m // (OUT_W * OUT_H)
-    
+
     # X strides for NHWC: stride_n=IN_H*IN_W*IN_C, stride_h=IN_W*IN_C, stride_w=IN_C, stride_c=1
     stride_xn = IN_H * IN_W * IN_C
     stride_xh = IN_W * IN_C
@@ -48,57 +53,48 @@

     stride_wo = KH * KW * IN_C
     stride_wkh = KW * IN_C
     stride_wkw = IN_C
-    
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-    
-    TOTAL_K = IN_C * KH * KW
-    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K
-    
-    for ijk in range(KH * KW * BLOCK_K_COUNT):
-        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
-        ij = ijk // BLOCK_K_COUNT
-        kh_idx = ij // KW
-        kw_idx = ij % KW
-        
-        offs_k = k + tl.arange(0, BLOCK_K)
-        # offs_k indexes into IN_C dimension (since we iterate kh, kw in outer loop)
-        in_c = offs_k  # within this kh, kw block
-        
-        ih = out_h * STRIDE_H + kh_idx - PAD_H
-        iw = out_w * STRIDE_W + kw_idx - PAD_W
-        
-        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (batch < BATCH)
-        valid_k = offs_k < IN_C
-        
-        # Load X tile [BLOCK_M, BLOCK_K]
-        x_ptrs = X_ptr + batch[:, None] * stride_xn + ih[:, None] * stride_xh + iw[:, None] * stride_xw + in_c[None, :]
-        x_mask = valid_spatial[:, None] & valid_k[None, :]
-        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
-        
-        # Load W tile [BLOCK_K, BLOCK_N] - W in OHWI layout
-        # W[out_c, kh_idx, kw_idx, in_c]
-        w_ptrs = W_ptr + offs_n[None, :] * stride_wo + kh_idx * stride_wkh + kw_idx * stride_wkw + offs_k[:, None]
-        w_mask = valid_k[:, None] & (offs_n[None, :] < OUT_C)
-        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
-        
-        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
-    
+
+    BLOCK_K_COUNT: tl.constexpr = (IN_C + BLOCK_K - 1) // BLOCK_K
+
+    for kh_idx in tl.static_range(KH):
+        for kw_idx in tl.static_range(KW):
+            ih = out_h * STRIDE_H + kh_idx - PAD_H
+            iw = out_w * STRIDE_W + kw_idx - PAD_W
+            valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (batch < BATCH)
+            for k_block in tl.static_range(BLOCK_K_COUNT):
+                offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
+                valid_k = offs_k < IN_C
+
+                # Load X tile [BLOCK_M, BLOCK_K]
+                x_ptrs = X_ptr + batch[:, None] * stride_xn + ih[:, None] * stride_xh + iw[:, None] * stride_xw + offs_k[None, :]
+                x_mask = valid_spatial[:, None] & valid_k[None, :]
+                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0, eviction_policy='evict_last')
+
+                # Load W tile [BLOCK_K, BLOCK_N] - W in OHWI layout
+                # W[out_c, kh_idx, kw_idx, in_c]
+                w_ptrs = W_ptr + offs_n[None, :] * stride_wo + kh_idx * stride_wkh + kw_idx * stride_wkw + offs_k[:, None]
+                w_mask = valid_k[:, None] & (offs_n[None, :] < OUT_C)
+                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0, eviction_policy='evict_last')
+
+                acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
+
     # Add bias, divide, leaky relu
-    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
+    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0, eviction_policy='evict_first')
     acc = acc + bias[None, :]
     acc = acc * (1.0 / DIVISOR)
     acc = tl.where(acc > 0.0, acc, acc * 0.01)
-    
+
     # Store output in NCHW layout: (BATCH, OUT_C, OUT_H, OUT_W)
     # stride_n=OUT_C*OUT_H*OUT_W, stride_c=OUT_H*OUT_W, stride_h=OUT_W, stride_w=1
     stride_yn = OUT_C * OUT_H * OUT_W
     stride_yc = OUT_H * OUT_W
     stride_yh = OUT_W
-    
+
     out_mask = (batch[:, None] < BATCH) & (out_h[:, None] < OUT_H) & (out_w[:, None] < OUT_W) & (offs_n[None, :] < OUT_C)
     y_ptrs = Y_ptr + batch[:, None] * stride_yn + offs_n[None, :] * stride_yc + out_h[:, None] * stride_yh + out_w[:, None]
-    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)
-
+    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask, eviction_policy='evict_first')
 
 def kernel_function(p_conv_weight, p_conv_bias, x):
     # x: (128, 8, 128, 128) NCHW
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
