# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_gelu_view_55` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9730 ns — **Best latency**: 9730 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_gelu_view_55`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_55_triton_poi_fused_addmm_gelu_vi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (11760,1,1), Waves/SM: 5.57
- **Registers/thread**: 16
- **Duration**: 9730 ns
- **SM Throughput**: 38.76 % | **Memory**: 16.84 % | **DRAM**: 12.68 %
- **Achieved Occupancy**: 47.56 %
- **IPC Active**: 2.05 inst/cycle | **No Eligible**: 47.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_pointwise_gelu | 0.996× | 0.203× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.760× | 0.710× | ✓ |  |
| C | autotune_xblock_for_latency_hiding | 1.011× | 1.000× | ✓ | **✓ best** |
| D | add_autotune_larger_xblock_vectorized_loads | 0.994× | 0.203× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ | autotune_xblock_f... | Δ | add_autotune_larg... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9730 ns | 48030 ns | +38300 ns (+393.6%) | 13700 ns | +3970 ns (+40.8%) | 9730 ns | +0 ns (+0.0%) | 48000 ns | +38270 ns (+393.3%) |
| SM Throughput (%) | 38.76 % | 81.15 % | +42.39 pp | 71.68 % | +32.92 pp | 38.87 % | +0.11 pp | 81.15 % | +42.39 pp |
| Memory Throughput (%) | 16.84 % | 7.50 % | -9.34 pp | 11.40 % | -5.44 pp | 16.93 % | +0.09 pp | 7.50 % | -9.34 pp |
| DRAM Throughput (%) | 12.68 % | 2.56 % | -10.12 pp | 8.99 % | -3.69 pp | 12.69 % | +0.01 pp | 2.56 % | -10.12 pp |
| Achieved Occupancy (%) | 47.56 % | 87.37 % | +39.81 pp | 83.15 % | +35.59 pp | 47.32 % | -0.24 pp | 87.33 % | +39.77 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 16.00 register/thread | +0 | 28.00 register/thread | +12 |
| Waves/SM | 5.57 | 22.27 | +16.70 | 5.57 | +0.00 | 5.57 | +0.00 | 22.27 | +16.70 |
| No Eligible (%) | 47.86 % | 15.43 % | -32.43 pp | 15.88 % | -31.98 pp | 47.93 % | +0.07 pp | 15.01 % | -32.85 pp |
| IPC Active | 2.05 inst/cycle | 3.39 inst/cycle | +1.34 | 3.33 inst/cycle | +1.28 | 2.04 inst/cycle | -0.01 | 3.39 inst/cycle | +1.34 |


### Key Learnings

#### ❌ Plan D: `add_autotune_larger_xblock_vectorized_loads` (0.203×)

Increasing XBLOCK via autotune from 128 to 512 raised register pressure from 16 to 28 registers/thread, which reduced the number of concurrent blocks per SM (Block Limit Registers dropped from 32 to 4). This forced the autotuner to select a 512-thread block, increasing waves_per_sm from 5.57 to 22.27 and total executed instructions from ~3M to ~31M (10x), causing duration to balloon from 9.73 µs to 48 µs despite improved occupancy (47% → 87%) and compute throughput (38% → 81%). The root cause is that the larger XBLOCK caused the kernel to do far more redundant work per element (likely due to loop unrolling or index recomputation overhead), and the register spill/pressure from 28 regs/thread severely limited block-level parallelism.

> **Applicable when**: Never apply this pattern as implemented: autotune with larger XBLOCK while keeping a fixed grid size (not using lambda meta grid). This is an implementation flaw that causes each program_id to process more elements than the grid covers, leading to massive instruction explosion.

> **Avoid when**: Do not add autotune with variable XBLOCK without also updating the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Do not increase XBLOCK when register count per thread is already near 16-20 and the kernel has complex index arithmetic, as register pressure will limit block occupancy. Do not apply when the kernel is already latency-bound with low memory throughput (<20%) — the bottleneck may not be addressable by XBLOCK alone.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_bias_gelu_pointwise | 1.002× | 1.003× | ✓ |  |
| B | persistent_tile_scheduling_for_pointwise_gelu | 1.006× | 0.997× | ✓ |  |
| C | fix_broken_autotune_grid_and_fuse_fp32_instructions | 0.998× | 0.997× | ✓ |  |
| D | fix_autotune_grid_bug_restore_fast_fixed_config | 1.013× | 1.003× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.995× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_broken_autotu... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9730 ns | 9700 ns | -30 ns (-0.3%) | 9760 ns | +30 ns (+0.3%) | 9760 ns | +30 ns (+0.3%) | 9700 ns | -30 ns (-0.3%) | 9860 ns | +130 ns (+1.3%) |
| SM Throughput (%) | 38.76 % | 41.88 % | +3.12 pp | 45.90 % | +7.14 pp | 38.76 % | +0.00 pp | 38.83 % | +0.07 pp | 38.23 % | -0.53 pp |
| Memory Throughput (%) | 16.84 % | 16.95 % | +0.11 pp | 16.86 % | +0.02 pp | 16.90 % | +0.06 pp | 16.88 % | +0.04 pp | 16.66 % | -0.18 pp |
| DRAM Throughput (%) | 12.68 % | 12.72 % | +0.04 pp | 12.65 % | -0.03 pp | 12.65 % | -0.03 pp | 12.69 % | +0.01 pp | 12.53 % | -0.15 pp |
| Achieved Occupancy (%) | 47.56 % | 48.52 % | +0.96 pp | 51.30 % | +3.74 pp | 46.81 % | -0.75 pp | 47.91 % | +0.35 pp | 48.14 % | +0.58 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 47.86 % | 45.97 % | -1.89 pp | 39.15 % | -8.71 pp | 48.15 % | +0.29 pp | 47.92 % | +0.06 pp | 48.15 % | +0.29 pp |
| IPC Active | 2.05 inst/cycle | 2.20 inst/cycle | +0.15 | 2.41 inst/cycle | +0.36 | 2.04 inst/cycle | -0.01 | 2.00 inst/cycle | -0.05 | 2.06 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0031× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_nonfused_fp32_instructions_into_fma_for_gelu | 1.001× | 0.987× | ✓ |  |
| B | cache_bias_in_shared_memory_to_eliminate_l1tex_stalls | 1.014× | 1.000× | ✓ | **✓ best** |
| C | fix_persistent_kernel_grid_to_num_sms | 0.994× | 0.984× | ✓ |  |
| D | preload_bias_into_registers_to_eliminate_l1tex_stall | 1.012× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_nonfused_fp3... | Δ | cache_bias_in_sha... | Δ | fix_persistent_ke... | Δ | preload_bias_into... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9730 ns | 9860 ns | +130 ns (+1.3%) | 9730 ns | +0 ns (+0.0%) | 9890 ns | +160 ns (+1.6%) | 9730 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 38.76 % | 41.81 % | +3.05 pp | 36.22 % | -2.54 pp | 45.90 % | +7.14 pp | 35.24 % | -3.52 pp |
| Memory Throughput (%) | 16.84 % | 16.65 % | -0.19 pp | 16.87 % | +0.03 pp | 16.53 % | -0.31 pp | 16.90 % | +0.06 pp |
| DRAM Throughput (%) | 12.68 % | 12.50 % | -0.18 pp | 12.68 % | +0.00 pp | 12.46 % | -0.22 pp | 12.69 % | +0.01 pp |
| Achieved Occupancy (%) | 47.56 % | 48.31 % | +0.75 pp | 46.36 % | -1.20 pp | 50.38 % | +2.82 pp | 46.61 % | -0.95 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 47.86 % | 44.19 % | -3.67 pp | 51.79 % | +3.93 pp | 38.83 % | -9.03 pp | 52.46 % | +4.60 pp |
| IPC Active | 2.05 inst/cycle | 2.16 inst/cycle | +0.11 | 1.89 inst/cycle | -0.16 | 2.41 inst/cycle | +0.36 | 1.86 inst/cycle | -0.19 |

---

## Iteration 4

**Best so far**: 1.0031× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_kernel_with_wider_warps_for_bias_reuse_and_latency_hiding | 0.995× | 1.000× | ✓ |  |
| B | eliminate_per_element_modulo_in_bias_load_for_coalesced_access | 1.013× | 1.003× | ✓ |  |
| C | fix_persistent_grid_to_num_sms_and_increase_num_warps | 1.000× | 0.991× | ✓ |  |
| D | fix_persistent_kernel_grid_launch_to_true_num_sms | 1.012× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | eliminate_per_ele... | Δ | fix_persistent_gr... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9730 ns | 9730 ns | +0 ns (+0.0%) | 9700 ns | -30 ns (-0.3%) | 9820 ns | +90 ns (+0.9%) | 9760 ns | +30 ns (+0.3%) |
| SM Throughput (%) | 38.76 % | 43.07 % | +4.31 pp | 36.07 % | -2.69 pp | 41.85 % | +3.09 pp | 45.93 % | +7.17 pp |
| Memory Throughput (%) | 16.84 % | 16.75 % | -0.09 pp | 16.71 % | -0.13 pp | 16.69 % | -0.15 pp | 16.78 % | -0.06 pp |
| DRAM Throughput (%) | 12.68 % | 12.69 % | +0.01 pp | 12.69 % | +0.01 pp | 12.55 % | -0.13 pp | 12.62 % | -0.06 pp |
| Achieved Occupancy (%) | 47.56 % | 48.43 % | +0.87 pp | 46.74 % | -0.82 pp | 48.83 % | +1.27 pp | 50.28 % | +2.72 pp |
| Registers/Thread | 16.00 register/thread | 17.00 register/thread | +1 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 47.86 % | 42.29 % | -5.57 pp | 51.50 % | +3.64 pp | 44.15 % | -3.71 pp | 39.03 % | -8.83 pp |
| IPC Active | 2.05 inst/cycle | 2.27 inst/cycle | +0.22 | 1.86 inst/cycle | -0.19 | 2.21 inst/cycle | +0.16 | 2.40 inst/cycle | +0.35 |

---

## Iteration 5

**Best so far**: 1.0031× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_xblock_to_bias_period_eliminate_modulo_and_reduce_cta_count | 1.000× | 1.003× | ✓ |  |
| B | fix_bias_index_modulo_and_true_persistent_grid | 1.013× | 1.000× | ✓ |  |
| C | fix_broken_persistent_grid_and_increase_num_warps_for_latency_hiding | 0.994× | 0.994× | ✓ |  |
| D | fix_persistent_kernel_true_sm_grid_with_wider_warps | 0.998× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_xblock_to_b... | Δ | fix_bias_index_mo... | Δ | fix_broken_persis... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9730 ns | 9700 ns | -30 ns (-0.3%) | 9730 ns | +0 ns (+0.0%) | 9790 ns | +60 ns (+0.6%) | 9700 ns | -30 ns (-0.3%) |
| SM Throughput (%) | 38.76 % | 32.85 % | -5.91 pp | 41.08 % | +2.32 pp | 45.94 % | +7.18 pp | 41.24 % | +2.48 pp |
| Memory Throughput (%) | 16.84 % | 16.35 % | -0.49 pp | 16.86 % | +0.02 pp | 16.67 % | -0.17 pp | 16.94 % | +0.10 pp |
| DRAM Throughput (%) | 12.68 % | 12.70 % | +0.02 pp | 12.69 % | +0.01 pp | 12.63 % | -0.05 pp | 12.71 % | +0.03 pp |
| Achieved Occupancy (%) | 47.56 % | 45.70 % | -1.86 pp | 49.35 % | +1.79 pp | 51.16 % | +3.60 pp | 48.01 % | +0.45 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 5.57 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 | 5.57 | +0.00 |
| No Eligible (%) | 47.86 % | 55.94 % | +8.08 pp | 43.95 % | -3.91 pp | 39.30 % | -8.56 pp | 43.91 % | -3.95 pp |
| IPC Active | 2.05 inst/cycle | 1.74 inst/cycle | -0.31 | 2.19 inst/cycle | +0.14 | 2.43 inst/cycle | +0.38 | 2.21 inst/cycle | +0.16 |

---

## Cumulative Progress

```
Baseline:  9730 ns  (1.000×)
Iter 2     9700 ns  (1.003×)  — fix_autotune_grid_bug_restore_fast_fixed_config

Final best: 9730 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_addmm_gelu_view_55(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1505280
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
-    x0 = (xindex % 3072)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    bias_start = xoffset % 3072  # scalar modulo, one op per CTA
+    bias_offsets = bias_start + tl.arange(0, XBLOCK)  # contiguous slice, no per-element modulo
+    bias_tile = tl.load(in_ptr0 + bias_offsets, xmask)
     tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp2 = tmp0 + tmp1
-    tmp3 = tl.full([1], 0.5, tl.float32)
-    tmp4 = tmp2 * tmp3
-    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
-    tmp6 = tmp2 * tmp5
-    tmp7 = libdevice.erf(tmp6)
-    tmp8 = tl.full([1], 1.0, tl.float32)
-    tmp9 = tmp7 + tmp8
-    tmp10 = tmp4 * tmp9
-    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
-
-
+    tmp2 = bias_tile + tmp1
+    erf_val = libdevice.erf(tmp2 * 0.7071067811865476)
+    result = tmp2 * (0.5 * erf_val + 0.5)
+    tl.store(in_out_ptr0 + (x2), result, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
