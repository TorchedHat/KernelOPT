# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 5920 ns (**0.9831×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 16.24 % | **Memory**: 42.45 % | **DRAM**: 21.15 %
- **Achieved Occupancy**: 54.62 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.257× | 0.983× | ✓ | **✓ best** |
| B | autotune_tile_sizes_and_warps_for_coalescing_and_occupancy | 1.219× | 1.433× | ✓ |  |
| D | increase_tile_sizes_and_autotune_for_underutilized_transpose_kernel | 0.933× | 0.418× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 5920 ns | +100 ns (+1.7%) | 4060 ns | -1760 ns (-30.2%) | 13920 ns | +8100 ns (+139.2%) |
| SM Throughput (%) | 16.24 % | 16.91 % | +0.67 pp | 17.00 % | +0.76 pp | 49.08 % | +32.84 pp |
| Memory Throughput (%) | 42.45 % | 41.64 % | -0.81 pp | 34.06 % | -8.39 pp | 60.51 % | +18.06 pp |
| DRAM Throughput (%) | 21.15 % | 20.76 % | -0.39 pp | 16.23 % | -4.92 pp | 8.85 % | -12.30 pp |
| Achieved Occupancy (%) | 54.62 % | 55.01 % | +0.39 pp | 53.59 % | -1.03 pp | 63.95 % | +9.33 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 21.00 register/thread | -5 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.98 | +1.24 |
| No Eligible (%) | 70.86 % | 70.51 % | -0.35 pp | 67.08 % | -3.78 pp | 45.02 % | -25.84 pp |
| IPC Active | 1.15 inst/cycle | 1.14 inst/cycle | -0.01 | 1.32 inst/cycle | +0.17 | 2.17 inst/cycle | +1.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads | 1.219× | 1.399× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints_and_pipelined_stages | 1.206× | 1.433× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 4160 ns | -1660 ns (-28.5%) | 4060 ns | -1760 ns (-30.2%) |
| SM Throughput (%) | 16.24 % | 16.76 % | +0.52 pp | 15.99 % | -0.25 pp |
| Memory Throughput (%) | 42.45 % | 33.45 % | -9.00 pp | 33.79 % | -8.66 pp |
| DRAM Throughput (%) | 21.15 % | 15.96 % | -5.19 pp | 16.16 % | -4.99 pp |
| Achieved Occupancy (%) | 54.62 % | 54.79 % | +0.17 pp | 54.23 % | -0.39 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.86 % | 67.50 % | -3.36 pp | 68.65 % | -2.21 pp |
| IPC Active | 1.15 inst/cycle | 1.34 inst/cycle | +0.19 | 1.34 inst/cycle | +0.19 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_bypass_with_lambda_grid | 1.227× | 1.357× | ✓ |  |
| B | fix_autotune_launch_site_and_channel_aligned_yblock_configs | 1.227× | 1.444× | ✓ |  |
| C | fix_autotune_grid_lambda_to_honor_tile_selection | 1.215× | 1.444× | ✓ |  |
| D | fix_autotune_grid_lambda_to_activate_tile_exploration | 1.221× | 1.444× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_bypa... | Δ | fix_autotune_laun... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 4290 ns | -1530 ns (-26.3%) | 4030 ns | -1790 ns (-30.8%) | 4030 ns | -1790 ns (-30.8%) | 4030 ns | -1790 ns (-30.8%) |
| SM Throughput (%) | 16.24 % | 17.25 % | +1.01 pp | 16.87 % | +0.63 pp | 16.55 % | +0.31 pp | 16.76 % | +0.52 pp |
| Memory Throughput (%) | 42.45 % | 32.20 % | -10.25 pp | 34.30 % | -8.15 pp | 33.94 % | -8.51 pp | 34.10 % | -8.35 pp |
| DRAM Throughput (%) | 21.15 % | 15.45 % | -5.70 pp | 16.39 % | -4.76 pp | 16.33 % | -4.82 pp | 16.38 % | -4.77 pp |
| Achieved Occupancy (%) | 54.62 % | 53.24 % | -1.38 pp | 52.74 % | -1.88 pp | 53.27 % | -1.35 pp | 55.11 % | +0.49 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 21.00 register/thread | -5 | 21.00 register/thread | -5 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.86 % | 64.80 % | -6.06 pp | 65.25 % | -5.61 pp | 65.48 % | -5.38 pp | 65.35 % | -5.51 pp |
| IPC Active | 1.15 inst/cycle | 1.32 inst/cycle | +0.17 | 1.33 inst/cycle | +0.18 | 1.32 inst/cycle | +0.17 | 1.21 inst/cycle | +0.06 |


### Key Learnings

#### ✅ Plan A: `fix_autotune_bypass_with_lambda_grid` (1.357×)

When a Triton kernel has a @triton.autotune decorator but the launch site uses a static grid tuple (e.g. `(cdiv(xnumel, 128), cdiv(ynumel, 8))`), the autotuner is completely bypassed and the kernel always runs with hardcoded tile sizes. Replacing the static tuple with a lambda grid (`lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))`) allows the autotuner to actually select configs at runtime. In this case the NCU profiling showed IssueSlotUtilization at 66% potential speedup and CPIStall at 37% due to L1TEX stalls, but the profiling metrics after the fix show only modest improvement in eligible warps (1.14→1.28) and no change in waves/SM (0.74), suggesting the autotuner selected a config that did not dramatically change tile geometry — the wall-clock speedup of 1.36x likely comes from end-to-end autotuning overhead amortization across repeated calls rather than a single-kernel NCU improvement.

> **Applicable when**: A Triton kernel has a @triton.autotune decorator with multiple tile configs (XBLOCK, YBLOCK, num_warps, num_stages variants) but the call site uses a static grid tuple with hardcoded tile sizes, meaning the autotuner decorator is present but never invoked. IssueSlotUtilization and CPIStall rules are triggered, suggesting suboptimal tile configuration.

> **Avoid when**: The autotune configs include large shared memory allocations (e.g. YBLOCK > 32 or num_stages > 4 with staging buffers) that could collapse Block Limit Shared Mem and cap theoretical occupancy — this was observed to cause 2.4x regression in a prior iteration. Also do not apply if the kernel has uncoalesced store patterns where larger tiles amplify per-CTA memory traffic penalty more than the autotuner benefit.

#### ✅ Plan C: `fix_autotune_grid_lambda_to_honor_tile_selection` (1.444×)

When a Triton kernel uses @triton.autotune but the launch site specifies a static grid tuple (e.g. `grid = (cdiv(xnumel, 128), cdiv(ynumel, 8))`), the autotuner's tile selection is silently ignored and the kernel always runs with the hardcoded tile sizes. Replacing the static grid with a lambda (`grid = lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK']))`) allows the autotuner to actually explore all configs and select the best-performing tile combination. In this case, the fix yielded a 1.44x speedup; the profiling metrics show modest but consistent improvements in eligible warps per scheduler (1.14→1.26), No Eligible cycles (68.65%→65.48%), and warp cycles per issued instruction (24.58→24.15), confirming the autotuner found a marginally better config once the grid lambda was corrected.

> **Applicable when**: A Triton kernel has @triton.autotune with multiple XBLOCK/YBLOCK configs but the launch site uses a static grid tuple with hardcoded tile sizes. IssueSlotUtilization and SOLBottleneck rules are triggered (waves/SM < 1.0, No Eligible > 60%), indicating the autotuner's tile exploration is being suppressed.

> **Avoid when**: The kernel intentionally uses a fixed tile size and does not use @triton.autotune, or when the grid is already a lambda reading META. Also avoid when autotune configs include large shared memory allocations that collapse Block Limit Shared Mem and cap theoretical occupancy (see autotune_larger_tiles_for_small_grid_transpose negative item).

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)

Final best: 5920 ns  (0.9831×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Small YBLOCK (4) + num_stages=3: low shared mem, pipelined
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8,  num_stages=3),
+        # Medium YBLOCK (8) + num_stages=3
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8,  num_stages=3),
+        # Larger YBLOCK (16/32) with num_stages=1 to avoid shared-mem blowup
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 16}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 32}, num_warps=8,  num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +33,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +42,13 @@

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
