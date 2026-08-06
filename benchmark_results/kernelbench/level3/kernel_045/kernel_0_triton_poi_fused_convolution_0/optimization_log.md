# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 4130 ns (**1.4649×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 16.57 % | **Memory**: 39.94 % | **DRAM**: 20.41 %
- **Achieved Occupancy**: 54.82 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.421× | 1.465× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_occupancy_and_latency_hiding | 1.275× | 1.027× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.278× | 1.401× | ✓ |  |
| D | autotune_tile_sizes_and_warps_for_underutilized_permutation_kernel | 1.206× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4130 ns | -1920 ns (-31.7%) | 5890 ns | -160 ns (-2.6%) | 4320 ns | -1730 ns (-28.6%) | 6020 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 16.57 % | 27.74 % | +11.17 pp | 16.65 % | +0.08 pp | 24.48 % | +7.91 pp | 16.54 % | -0.03 pp |
| Memory Throughput (%) | 39.94 % | 22.05 % | -17.89 pp | 42.84 % | +2.90 pp | 21.57 % | -18.37 pp | 41.05 % | +1.11 pp |
| DRAM Throughput (%) | 20.41 % | 4.01 % | -16.40 pp | 20.93 % | +0.52 pp | 7.16 % | -13.25 pp | 20.49 % | +0.08 pp |
| Achieved Occupancy (%) | 54.82 % | 67.87 % | +13.05 pp | 55.53 % | +0.71 pp | 53.51 % | -1.31 pp | 55.27 % | +0.45 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 28.00 register/thread | +2 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.67 % | 45.00 % | -25.67 pp | 71.94 % | +1.27 pp | 40.64 % | -30.03 pp | 70.68 % | +0.01 pp |
| IPC Active | 1.15 inst/cycle | 2.39 inst/cycle | +1.24 | 1.13 inst/cycle | -0.02 | 2.38 inst/cycle | +1.23 | 1.13 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.4649× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_via_compiler_hints | 1.235× | 1.401× | ✓ |  |
| C | vectorized_loads_with_coalesced_store_axis_swap | 1.149× | 0.856× | ✓ |  |
| D | vectorized_loads_with_max_contiguous_hints | 1.327× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4320 ns | -1730 ns (-28.6%) | 7070 ns | +1020 ns (+16.9%) | 5920 ns | -130 ns (-2.1%) |
| SM Throughput (%) | 16.57 % | 24.26 % | +7.69 pp | 19.84 % | +3.27 pp | 16.35 % | -0.22 pp |
| Memory Throughput (%) | 39.94 % | 21.32 % | -18.62 pp | 36.75 % | -3.19 pp | 42.00 % | +2.06 pp |
| DRAM Throughput (%) | 20.41 % | 7.18 % | -13.23 pp | 9.28 % | -11.13 pp | 20.86 % | +0.45 pp |
| Achieved Occupancy (%) | 54.82 % | 53.60 % | -1.22 pp | 50.75 % | -4.07 pp | 55.05 % | +0.23 pp |
| Registers/Thread | 26.00 register/thread | 28.00 register/thread | +2 | 32.00 register/thread | +6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.67 % | 41.04 % | -29.63 pp | 65.20 % | -5.47 pp | 72.38 % | +1.71 pp |
| IPC Active | 1.15 inst/cycle | 2.33 inst/cycle | +1.18 | 1.33 inst/cycle | +0.18 | 1.11 inst/cycle | -0.04 |

---

## Iteration 3

**Best so far**: 1.4649× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_add_flat_large_block_configs | 1.216× | 1.420× | ✓ |  |
| C | flatten_2d_to_1d_grid_with_large_xblock_for_full_vectorization | 1.304× | 0.248× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | flatten_2d_to_1d_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 4260 ns | -1790 ns (-29.6%) | 24350 ns | +18300 ns (+302.5%) |
| SM Throughput (%) | 16.57 % | 27.85 % | +11.28 pp | 17.82 % | +1.25 pp |
| Memory Throughput (%) | 39.94 % | 22.12 % | -17.82 pp | 77.77 % | +37.83 pp |
| DRAM Throughput (%) | 20.41 % | 3.88 % | -16.53 pp | 5.08 % | -15.33 pp |
| Achieved Occupancy (%) | 54.82 % | 68.63 % | +13.81 pp | 43.96 % | -10.86 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 56.00 register/thread | +30 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 2.97 | +2.23 |
| No Eligible (%) | 70.67 % | 44.13 % | -26.54 pp | 79.19 % | +8.52 pp |
| IPC Active | 1.15 inst/cycle | 2.40 inst/cycle | +1.25 | 0.82 inst/cycle | -0.33 |


### Key Learnings

#### ❌ Plan C: `flatten_2d_to_1d_grid_with_large_xblock_for_full_vectorization` (0.248×)

Replacing a 2D (XBLOCK×YBLOCK) tile layout with a 1D flat index over all elements caused a 4x regression. The flat index computation (modulo and integer division by xnumel=50176) introduced heavy arithmetic that more than doubled register usage from 28 to 56 registers/thread, cutting theoretical occupancy from 100% to 50% and reducing achieved occupancy from 53.6% to 43.96%. The instruction count exploded from 817K to 3.4M (4.15x), and eligible warps per scheduler collapsed from 2.61 to 0.51, worsening IssueSlotUtilization from 41% no-eligible to 79% no-eligible. Although waves/SM increased from 0.74 to 2.97, the register-pressure-induced occupancy cap and massive instruction overhead dominated, causing duration to increase from 4.32µs to 24.35µs. The L1 hit rate also dropped from 45.87% to 1.25% because the flat index destroyed the spatial locality that the 2D tile layout exploited.

> **Applicable when**: Never: this transformation is harmful when the flat index requires expensive modulo/division by a large non-power-of-2 constant (e.g., xnumel=50176), as it causes register explosion and destroys L1 spatial locality.

> **Avoid when**: Always avoid flattening a 2D grid to 1D when: (1) the x-dimension is a large non-power-of-2 value requiring expensive integer division/modulo to recover 2D coordinates, (2) the kernel already has moderate register usage (>20 registers/thread) that would be pushed over the occupancy-limiting threshold by the added arithmetic, (3) the 2D tile layout already achieves reasonable L1 hit rates (>30%) by exploiting spatial locality. The waves/SM improvement from flattening does not compensate for register-pressure-induced occupancy collapse and instruction count explosion.

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 1     4130 ns  (1.465×)  — vectorized_loads_with_autotune

Final best: 4130 ns  (1.4649×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=4),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4),
+        triton.Config({'YBLOCK': 16, 'XBLOCK': 128}, num_warps=4),
+        triton.Config({'YBLOCK': 32, 'XBLOCK': 128}, num_warps=8),
+        triton.Config({'YBLOCK': 8,  'XBLOCK': 256}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
     xnumel = 50176
+    xoffset = tl.program_id(0) * XBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # slow axis
+    xmask = xindex < xnumel
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]  # fast axis - stride-1 in output
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 3)
+    y0 = (yindex % 3)   # stride-1 in output, now fast-varying across threads
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    x2_vec = tl.max_contiguous(tl.multiple_of(x2, [XBLOCK, 1]), [XBLOCK, 1])
+    tmp0 = tl.load(in_ptr0 + (x2_vec + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +42,7 @@

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
