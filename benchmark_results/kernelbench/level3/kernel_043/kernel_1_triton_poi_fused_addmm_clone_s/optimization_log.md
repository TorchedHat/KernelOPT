# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 243870 ns — **Best latency**: 243870 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_1_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 243870 ns
- **SM Throughput**: 34.46 % | **Memory**: 32.05 % | **DRAM**: 32.05 %
- **Achieved Occupancy**: 43.73 %
- **IPC Active**: 1.39 inst/cycle | **No Eligible**: 65.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_larger_xblock_to_reduce_launch_overhead_and_improve_occupancy | 0.792× | 0.799× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 0.876× | 0.886× | ✓ |  |
| C | increase_xblock_with_autotune | 0.876× | 0.536× | ✓ |  |
| D | vectorized_loads_larger_xblock_autotune | 0.877× | 0.889× | ✓ |  |
| E | add_triton_autotune | 0.998× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_larger_x... | Δ | autotune_xblock_i... | Δ | increase_xblock_w... | Δ | vectorized_loads_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243870 ns | 305120 ns | +61250 ns (+25.1%) | 275140 ns | +31270 ns (+12.8%) | 454880 ns | +211010 ns (+86.5%) | 274210 ns | +30340 ns (+12.4%) | 243970 ns | +100 ns (+0.0%) |
| SM Throughput (%) | 34.46 % | 52.78 % | +18.32 pp | 37.44 % | +2.98 pp | 71.22 % | +36.76 pp | 37.92 % | +3.46 pp | 36.66 % | +2.20 pp |
| Memory Throughput (%) | 32.05 % | 25.31 % | -6.74 pp | 28.12 % | -3.93 pp | 26.88 % | -5.17 pp | 28.22 % | -3.83 pp | 32.03 % | -0.02 pp |
| DRAM Throughput (%) | 32.05 % | 25.31 % | -6.74 pp | 28.12 % | -3.93 pp | 16.98 % | -15.07 pp | 28.22 % | -3.83 pp | 32.03 % | -0.02 pp |
| Achieved Occupancy (%) | 43.73 % | 36.19 % | -7.54 pp | 35.77 % | -7.96 pp | 81.94 % | +38.21 pp | 35.47 % | -8.26 pp | 44.31 % | +0.58 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 18.00 register/thread | +2 | 28.00 register/thread | +12 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 372.36 | +186.18 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 65.22 % | 46.37 % | -18.85 pp | 60.40 % | -4.82 pp | 28.83 % | -36.39 pp | 60.63 % | -4.59 pp | 63.01 % | -2.21 pp |
| IPC Active | 1.39 inst/cycle | 2.19 inst/cycle | +0.80 | 1.59 inst/cycle | +0.20 | 2.84 inst/cycle | +1.45 | 1.54 inst/cycle | +0.15 | 1.49 inst/cycle | +0.10 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_with_autotune` (0.536×)

Adding a @triton.autotune decorator to increase XBLOCK (from 128 to 256 as selected) caused registers per thread to jump from 16 to 28, and executed instructions to balloon from 66M to 249M (3.76x more), resulting in a 1.87x longer kernel duration despite improved occupancy (43.7% → 81.9%) and scheduler utilization (34.5% → 71.2%). The autotune also failed to update the launch grid (it remained fixed at cdiv(xnumel, 128) instead of using a lambda), meaning the grid was not reduced proportionally to the larger XBLOCK, causing each element to be processed multiple times and inflating instruction count. When adding autotune to a pointwise kernel, the launch grid must be updated to a lambda using meta['XBLOCK'] to avoid redundant work; a fixed grid with a larger block size is strictly worse.

> **Applicable when**: Never apply this exact pattern (autotune with fixed grid). The intended strategy (larger XBLOCK + lambda grid) may be valid when: achieved occupancy < 50%, no-eligible-warp % > 60%, L1TEX stall dominates CPI, kernel is purely pointwise with low register pressure (< 20 regs/thread), and the grid is large (> 100K blocks).

> **Avoid when**: Do not add @triton.autotune to increase XBLOCK without simultaneously converting the launch grid to a lambda that uses meta['XBLOCK']. If the grid remains fixed at the original small-XBLOCK size, each output element will be written multiple times, causing instruction count to multiply by the XBLOCK ratio and severely degrading performance. Also avoid when register pressure is already moderate (≥ 20 regs/thread), as larger blocks will cause register spilling.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 1.000× | 0.999× | ✓ |  |
| B | persistent_tile_scheduling | 1.000× | 1.000× | ✓ |  |
| C | persistent_tile_scheduling | 1.000× | 1.000× | ✓ |  |
| D | persistent_tile_scheduling | 1.000× | 1.000× | ✓ |  |
| E | add_triton_autotune | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 243870 ns | 244130 ns | +260 ns (+0.1%) | 244000 ns | +130 ns (+0.1%) | 244000 ns | +130 ns (+0.1%) | 244000 ns | +130 ns (+0.1%) | 244220 ns | +350 ns (+0.1%) |
| SM Throughput (%) | 34.46 % | 48.01 % | +13.55 pp | 42.32 % | +7.86 pp | 47.20 % | +12.74 pp | 42.63 % | +8.17 pp | 36.86 % | +2.40 pp |
| Memory Throughput (%) | 32.05 % | 32.01 % | -0.04 pp | 32.03 % | -0.02 pp | 32.02 % | -0.03 pp | 32.03 % | -0.02 pp | 32.00 % | -0.05 pp |
| DRAM Throughput (%) | 32.05 % | 32.01 % | -0.04 pp | 32.03 % | -0.02 pp | 32.02 % | -0.03 pp | 32.03 % | -0.02 pp | 32.00 % | -0.05 pp |
| Achieved Occupancy (%) | 43.73 % | 48.02 % | +4.29 pp | 46.32 % | +2.59 pp | 46.60 % | +2.87 pp | 46.33 % | +2.60 pp | 44.16 % | +0.43 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 65.22 % | 51.20 % | -14.02 pp | 56.98 % | -8.24 pp | 52.08 % | -13.14 pp | 57.24 % | -7.98 pp | 63.01 % | -2.21 pp |
| IPC Active | 1.39 inst/cycle | 1.94 inst/cycle | +0.55 | 1.71 inst/cycle | +0.32 | 1.91 inst/cycle | +0.52 | 1.71 inst/cycle | +0.32 | 1.48 inst/cycle | +0.09 |

---

## Cumulative Progress

```
Baseline:  243870 ns  (1.000×)

Final best: 243870 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,23 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_clone_split_transpose_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 96)
     x2 = ((xindex // 49152) % 8)
     x1 = ((xindex // 96) % 512)
     x3 = xindex // 393216
     x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
+    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (x0 + 96*x2 + 2304*x1 + 1179648*x3), xmask)
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x4), tmp2, None)
-
-
+    tl.store(out_ptr0 + (x4), tmp2, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -29,7 +37,7 @@

     arg_2 = inputs[2]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_clone_split_transpose_view_1[grid](arg_0, arg_1, arg_2, 50331648, XBLOCK=128)
+    triton_poi_fused_addmm_clone_split_transpose_view_1[grid](arg_0, arg_1, arg_2, 50331648)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
