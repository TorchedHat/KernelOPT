# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_39` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 20670 ns — **Best latency**: 20670 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_39`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_39_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (29400,1,1), Waves/SM: 13.92
- **Registers/thread**: 22
- **Duration**: 20670 ns
- **SM Throughput**: 62.05 % | **Memory**: 23.20 % | **DRAM**: 15.10 %
- **Achieved Occupancy**: 61.09 %
- **IPC Active**: 2.77 inst/cycle | **No Eligible**: 29.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.854× | 0.678× | ✓ |  |
| B | autotune_xblock_num_warps_to_improve_occupancy_and_hide_l1tex_latency | 0.992× | 0.890× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_reduced_l1tex_stalls | 0.847× | 0.683× | ✓ |  |
| D | autotune_xblock_num_warps_for_occupancy_and_l1_reuse | 0.999× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20670 ns | 30500 ns | +9830 ns (+47.6%) | 23230 ns | +2560 ns (+12.4%) | 30240 ns | +9570 ns (+46.3%) | 20740 ns | +70 ns (+0.3%) |
| SM Throughput (%) | 62.05 % | 79.53 % | +17.48 pp | 72.49 % | +10.44 pp | 79.46 % | +17.41 pp | 60.87 % | -1.18 pp |
| Memory Throughput (%) | 23.20 % | 22.89 % | -0.31 pp | 26.95 % | +3.75 pp | 22.72 % | -0.48 pp | 22.75 % | -0.45 pp |
| DRAM Throughput (%) | 15.10 % | 10.31 % | -4.79 pp | 13.51 % | -1.59 pp | 10.38 % | -4.72 pp | 15.14 % | +0.04 pp |
| Achieved Occupancy (%) | 61.09 % | 87.39 % | +26.30 pp | 71.33 % | +10.24 pp | 87.21 % | +26.12 pp | 61.59 % | +0.50 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 26.00 register/thread | +4 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 13.92 | 13.92 | +0.00 | 13.92 | +0.00 | 13.92 | +0.00 | 13.92 | +0.00 |
| No Eligible (%) | 29.82 % | 14.58 % | -15.24 pp | 17.54 % | -12.28 pp | 14.66 % | -15.16 pp | 29.85 % | +0.03 pp |
| IPC Active | 2.77 inst/cycle | 3.41 inst/cycle | +0.64 | 3.26 inst/cycle | +0.49 | 3.41 inst/cycle | +0.64 | 2.76 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.678×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that had low achieved occupancy (61%) and high L1TEX stall (39.8%) improved occupancy to 87% and eligible warps/scheduler from 1.98 to 7.28, but caused a 47% increase in kernel duration (20.67 µs → 30.5 µs). The root cause is that the selected larger XBLOCK config increased register pressure from 22 to 32 registers/thread and nearly doubled the total executed instruction count (9.99M → 18.96M), indicating the autotune selected a configuration that introduced significant instruction overhead without proportional throughput gain. When the kernel is already compute-bound (62% SM throughput) and the instruction count grows faster than the latency hiding benefit, larger tiles cause net regression despite improved occupancy metrics.

> **Applicable when**: Kernel has low achieved occupancy (<65%) AND low eligible warps/scheduler (<2.0) AND is memory-latency bound (high L1TEX stall %) AND compute SM throughput is moderate (<50%) — meaning there is genuine idle compute capacity to absorb more work per block.

> **Avoid when**: Kernel is already compute-bound (SM throughput >60%) even at low occupancy; when larger XBLOCK causes register spill or significant register pressure increase (e.g., 22→32 regs/thread); when the kernel body has enough arithmetic that doubling tile size doubles instruction count without proportional memory access reduction; when the bottleneck is compute throughput rather than memory latency hiding.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_and_add_num_warps16_for_latency_hiding | 0.999× | 0.988× | ✓ |  |
| B | fuse_fp32_arithmetic_to_fma_in_batchnorm_relu_kernel | 0.998× | 0.986× | ✓ |  |
| C | reduce_non_fused_fp32_instructions_via_arithmetic_simplification | 0.998× | 1.003× | ✓ |  |
| D | reduce_non_fused_fp32_instructions_via_explicit_fma | 0.998× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fuse_fp32_arithme... | Δ | reduce_non_fused_... | Δ | reduce_non_fused_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 20670 ns | 20930 ns | +260 ns (+1.3%) | 20960 ns | +290 ns (+1.4%) | 20610 ns | -60 ns (-0.3%) | 20900 ns | +230 ns (+1.1%) |
| SM Throughput (%) | 62.05 % | 45.87 % | -16.18 pp | 46.44 % | -15.61 pp | 46.90 % | -15.15 pp | 46.87 % | -15.18 pp |
| Memory Throughput (%) | 23.20 % | 20.55 % | -2.65 pp | 30.82 % | +7.62 pp | 22.96 % | -0.24 pp | 22.92 % | -0.28 pp |
| DRAM Throughput (%) | 15.10 % | 14.93 % | -0.17 pp | 14.99 % | -0.11 pp | 15.23 % | +0.13 pp | 15.08 % | -0.02 pp |
| Achieved Occupancy (%) | 61.09 % | 38.86 % | -22.23 pp | 32.98 % | -28.11 pp | 43.65 % | -17.44 pp | 43.59 % | -17.50 pp |
| Registers/Thread | 22.00 register/thread | 26.00 register/thread | +4 | 26.00 register/thread | +4 | 20.00 register/thread | -2 | 20.00 register/thread | -2 |
| Waves/SM | 13.92 | 6.96 | -6.96 | 13.92 | +0.00 | 13.92 | +0.00 | 13.92 | +0.00 |
| No Eligible (%) | 29.82 % | 44.74 % | +14.92 pp | 48.69 % | +18.87 pp | 56.70 % | +26.88 pp | 55.77 % | +25.95 pp |
| IPC Active | 2.77 inst/cycle | 2.17 inst/cycle | -0.60 | 2.02 inst/cycle | -0.75 | 1.75 inst/cycle | -1.02 | 1.75 inst/cycle | -1.02 |

---

## Cumulative Progress

```
Baseline:  20670 ns  (1.000×)

Final best: 20670 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=8),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3763200
@@ -33,8 +47,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +56,7 @@

     arg_5 = inputs[5]
     xnumel = 3763200
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_39[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3763200, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_39[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3763200)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
