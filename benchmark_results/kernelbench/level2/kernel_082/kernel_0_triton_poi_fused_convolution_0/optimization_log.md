# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 28480 ns (**0.2100×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_082/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.10 % | **Memory**: 41.02 % | **DRAM**: 20.59 %
- **Achieved Occupancy**: 54.88 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.061× | 1.191× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.340× | 0.210× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_underutilized_transpose_kernel | 1.308× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 5020 ns | -960 ns (-16.1%) | 28480 ns | +22500 ns (+376.3%) | 5890 ns | -90 ns (-1.5%) |
| SM Throughput (%) | 16.10 % | 21.46 % | +5.36 pp | 60.86 % | +44.76 pp | 16.35 % | +0.25 pp |
| Memory Throughput (%) | 41.02 % | 28.02 % | -13.00 pp | 73.87 % | +32.85 pp | 42.41 % | +1.39 pp |
| DRAM Throughput (%) | 20.59 % | 13.16 % | -7.43 pp | 4.32 % | -16.27 pp | 20.89 % | +0.30 pp |
| Achieved Occupancy (%) | 54.88 % | 47.15 % | -7.73 pp | 34.31 % | -20.57 pp | 55.37 % | +0.49 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 32.00 register/thread | +6 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 3.96 | +3.22 | 0.74 | +0.00 |
| No Eligible (%) | 71.06 % | 55.74 % | -15.32 pp | 48.28 % | -22.78 pp | 70.89 % | -0.17 pp |
| IPC Active | 1.14 inst/cycle | 1.70 inst/cycle | +0.56 | 2.05 inst/cycle | +0.91 | 1.14 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_yblock_aligned_to_channel_stride_3 | 1.007× | 0.926× | ✓ |  |
| C | fix_autotune_yblock_stride3_alignment_and_evict_first | 1.317× | 0.206× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_yblock_a... | Δ | fix_autotune_yblo... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 6460 ns | +480 ns (+8.0%) | 29020 ns | +23040 ns (+385.3%) |
| SM Throughput (%) | 16.10 % | 36.50 % | +20.40 pp | 60.53 % | +44.43 pp |
| Memory Throughput (%) | 41.02 % | 56.50 % | +15.48 pp | 73.49 % | +32.47 pp |
| DRAM Throughput (%) | 20.59 % | 10.19 % | -10.40 pp | 4.24 % | -16.35 pp |
| Achieved Occupancy (%) | 54.88 % | 44.11 % | -10.77 pp | 34.31 % | -20.57 pp |
| Registers/Thread | 26.00 register/thread | 53.00 register/thread | +27 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.32 | +0.58 | 3.96 | +3.22 |
| No Eligible (%) | 71.06 % | 46.05 % | -25.01 pp | 48.70 % | -22.36 pp |
| IPC Active | 1.14 inst/cycle | 2.13 inst/cycle | +0.99 | 2.05 inst/cycle | +0.91 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 1.077× | 1.068× | ✓ |  |
| B | fix_grid_lambda_to_use_autotuned_tile_sizes | 1.016× | 0.890× | ✓ |  |
| C | reduce_autotune_smem_overhead_with_stride3_aligned_yblock | 1.119× | 1.280× | ✓ |  |
| D | trim_autotune_configs_to_stride3_multiples_restore_occupancy | 1.222× | 1.289× | ✓ |  |
| E | add_triton_autotune | 0.905× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | fix_grid_lambda_t... | Δ | reduce_autotune_s... | Δ | trim_autotune_con... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 5600 ns | -380 ns (-6.4%) | 6720 ns | +740 ns (+12.4%) | 4670 ns | -1310 ns (-21.9%) | 4640 ns | -1340 ns (-22.4%) | 5980 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 16.10 % | 25.51 % | +9.41 pp | 36.57 % | +20.47 pp | 20.95 % | +4.85 pp | 20.90 % | +4.80 pp | 16.23 % | +0.13 pp |
| Memory Throughput (%) | 41.02 % | 35.47 % | -5.55 pp | 56.68 % | +15.66 pp | 30.24 % | -10.78 pp | 30.45 % | -10.57 pp | 41.51 % | +0.49 pp |
| DRAM Throughput (%) | 20.59 % | 11.79 % | -8.80 pp | 9.81 % | -10.78 pp | 14.15 % | -6.44 pp | 14.22 % | -6.37 pp | 20.60 % | +0.01 pp |
| Achieved Occupancy (%) | 54.88 % | 52.92 % | -1.96 pp | 44.47 % | -10.41 pp | 46.50 % | -8.38 pp | 47.68 % | -7.20 pp | 55.48 % | +0.60 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 53.00 register/thread | +27 | 27.00 register/thread | +1 | 27.00 register/thread | +1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.32 | +0.58 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.06 % | 52.72 % | -18.34 pp | 46.00 % | -25.06 pp | 55.51 % | -15.55 pp | 56.14 % | -14.92 pp | 70.05 % | -1.01 pp |
| IPC Active | 1.14 inst/cycle | 1.86 inst/cycle | +0.72 | 2.08 inst/cycle | +0.94 | 1.69 inst/cycle | +0.55 | 1.74 inst/cycle | +0.60 | 1.14 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan A: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (1.068×)

When a Triton kernel's register count per thread (e.g., 53) limits Block Limit Registers well below Block Limit Warps (9 vs. 16), adding maxnreg=32 to every autotune Config forces ptxas to spill excess registers to local memory, raising Block Limit Registers to match Block Limit Warps and pushing theoretical occupancy from ~56% to 100%. In this case, the register reduction also cut total executed instructions by ~41% (1.83M → 1.08M), contributing to a 13% wall-time reduction (6.46 µs → 5.60 µs) despite a slight increase in 'No Eligible' stall percentage. The strategy is most effective when the kernel is latency-bound with many idle scheduler cycles and register pressure is the primary occupancy limiter.

> **Applicable when**: Triton kernel where registers_per_thread is high (>40) and Block Limit Registers < Block Limit Warps, causing theoretical occupancy < 75%. The kernel is latency-bound with significant 'No Eligible' scheduler stall cycles (>40%). TheoreticalOccupancy and/or LaunchConfiguration NCU rules are triggered. The kernel body is compute-light enough that register spilling to L1 does not dominate runtime.

> **Avoid when**: Kernel is already compute-bound (SM Throughput > 80%) — register spilling adds instruction overhead that worsens throughput. Block Limit Registers already equals Block Limit Warps (register pressure is not the occupancy limiter). The kernel has heavy reuse of register-resident values across many instructions where spilling would cause repeated L1 round-trips.

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)

Final best: 28480 ns  (0.2100×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=16),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 32}, num_warps=16),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

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
