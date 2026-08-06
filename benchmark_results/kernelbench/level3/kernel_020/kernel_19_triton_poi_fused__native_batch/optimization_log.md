# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11870 ns — **Best latency**: 20380 ns (**0.5824×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_19_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (14700,1,1), Waves/SM: 6.96
- **Registers/thread**: 18
- **Duration**: 11870 ns
- **SM Throughput**: 55.78 % | **Memory**: 21.49 % | **DRAM**: 12.96 %
- **Achieved Occupancy**: 60.92 %
- **IPC Active**: 2.76 inst/cycle | **No Eligible**: 31.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.790× | 0.729× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.976× | 0.966× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.776× | 0.729× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy | 1.015× | 0.582× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11870 ns | 16290 ns | +4420 ns (+37.2%) | 12290 ns | +420 ns (+3.5%) | 16290 ns | +4420 ns (+37.2%) | 20380 ns | +8510 ns (+71.7%) | 11870 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 55.78 % | 73.38 % | +17.60 pp | 44.69 % | -11.09 pp | 73.23 % | +17.45 pp | 75.45 % | +19.67 pp | 55.81 % | +0.03 pp |
| Memory Throughput (%) | 21.49 % | 22.04 % | +0.55 pp | 31.37 % | +9.88 pp | 21.99 % | +0.50 pp | 24.45 % | +2.96 pp | 21.49 % | +0.00 pp |
| DRAM Throughput (%) | 12.96 % | 9.43 % | -3.53 pp | 12.49 % | -0.47 pp | 9.44 % | -3.52 pp | 7.54 % | -5.42 pp | 12.98 % | +0.02 pp |
| Achieved Occupancy (%) | 60.92 % | 85.09 % | +24.17 pp | 42.66 % | -18.26 pp | 84.61 % | +23.69 pp | 82.87 % | +21.95 pp | 60.80 % | -0.12 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 18.00 register/thread | +0 |
| Waves/SM | 6.96 | 6.96 | +0.00 | 6.96 | +0.00 | 6.96 | +0.00 | 13.92 | +6.96 | 6.96 | +0.00 |
| No Eligible (%) | 31.01 % | 16.28 % | -14.73 pp | 41.97 % | +10.96 pp | 16.16 % | -14.85 pp | 15.67 % | -15.34 pp | 31.22 % | +0.21 pp |
| IPC Active | 2.76 inst/cycle | 3.29 inst/cycle | +0.53 | 2.30 inst/cycle | -0.46 | 3.33 inst/cycle | +0.57 | 3.36 inst/cycle | +0.60 | 2.74 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.729×)

Adding triton.autotune with larger XBLOCK configs (512–2048) increased registers per thread from 18 to 32, nearly doubling executed instructions (5.1M → 9.3M) and increasing kernel duration by 37% despite higher occupancy. The grid was left as a fixed tuple (cdiv(xnumel, 128)) instead of a lambda, so the larger XBLOCK configs still launched with the original small grid, causing each CTA to process more elements than the grid covered — this is an implementation flaw that also inflated instruction count. Even if the grid were fixed, the register pressure increase from 18 to 32 registers/thread reduced Block Limit Registers from 21 to 16 blocks/SM, and the instruction count nearly doubled, suggesting the autotune selected a config that unrolled aggressively and generated far more instructions than the original.

> **Applicable when**: Never apply this pattern as implemented; the grid must be a lambda when using autotune with variable XBLOCK

> **Avoid when**: Do not add triton.autotune with larger XBLOCK configs when: (1) the grid is a fixed tuple rather than a lambda — this causes incorrect coverage and instruction bloat; (2) the kernel has low register count (≤18) but the larger tile configs cause register count to jump significantly (e.g. 18→32), nearly doubling instruction count; (3) the kernel is already compute-bound at low register pressure — larger tiles may increase IPC but also increase total instructions, resulting in longer duration

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_occupancy` (0.582×)

Doubling block size from 128 to 256 (via autotune selecting XBLOCK=256) increased achieved occupancy from 60.9% to 82.9% and eligible warps per scheduler from 1.89 to 6.5, but total executed instructions ballooned from 5.1M to 12.2M (2.4x increase) and registers per thread rose from 18 to 24, causing the kernel duration to increase from 11.87µs to 20.38µs (0.58x speedup). The autotune grid was incorrectly left as a fixed `cdiv(xnumel, 128)` rather than being updated to `lambda meta: cdiv(xnumel, meta['XBLOCK'])`, meaning the grid size did not shrink proportionally with the larger block — each element was processed multiple times, causing redundant work. When using autotune to sweep block sizes, the launch grid MUST be a lambda that reads `meta['XBLOCK']`; a static grid computed from the old block size will over-launch CTAs and multiply instruction count.

> **Applicable when**: Any time triton.autotune is added to sweep XBLOCK/tile sizes; the grid must always be a lambda consuming meta['XBLOCK'] so CTA count scales inversely with block size.

> **Avoid when**: Fixed-grid launches where XBLOCK is a compile-time constant passed explicitly — in that case a static grid is correct. Do not use a static grid with autotune.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_div_with_rsqrt_to_enable_fma_fusion | 1.015× | 1.025× | ✓ |  |
| B | apply_evict_first_to_streaming_tensor_load_and_store | 0.960× | 0.964× | ✓ |  |
| C | fuse_non_fused_fp32_instructions_into_fma | 0.960× | 0.961× | ✓ |  |
| D | rewrite_bn_arithmetic_to_enable_fma_fusion | 0.961× | 0.959× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_div_... | Δ | apply_evict_first... | Δ | fuse_non_fused_fp... | Δ | rewrite_bn_arithm... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11870 ns | 11580 ns | -290 ns (-2.4%) | 12320 ns | +450 ns (+3.8%) | 12350 ns | +480 ns (+4.0%) | 12380 ns | +510 ns (+4.3%) |
| SM Throughput (%) | 55.78 % | 33.63 % | -22.15 pp | 47.10 % | -8.68 pp | 44.66 % | -11.12 pp | 44.73 % | -11.05 pp |
| Memory Throughput (%) | 21.49 % | 21.35 % | -0.14 pp | 31.61 % | +10.12 pp | 31.17 % | +9.68 pp | 31.35 % | +9.86 pp |
| DRAM Throughput (%) | 12.96 % | 13.27 % | +0.31 pp | 12.47 % | -0.49 pp | 12.43 % | -0.53 pp | 12.41 % | -0.55 pp |
| Achieved Occupancy (%) | 60.92 % | 45.33 % | -15.59 pp | 44.35 % | -16.57 pp | 42.54 % | -18.38 pp | 42.85 % | -18.07 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 6.96 | 6.96 | +0.00 | 6.96 | +0.00 | 6.96 | +0.00 | 6.96 | +0.00 |
| No Eligible (%) | 31.01 % | 56.58 % | +25.57 pp | 38.92 % | +7.91 pp | 42.66 % | +11.65 pp | 42.69 % | +11.68 pp |
| IPC Active | 2.76 inst/cycle | 1.71 inst/cycle | -1.05 | 2.40 inst/cycle | -0.36 | 2.31 inst/cycle | -0.45 | 2.31 inst/cycle | -0.45 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_to_lambda_for_correct_xblock_scaling | 1.008× | 1.005× | ✓ |  |
| B | precompute_bn_scale_bias_to_expose_fma_fusion | 0.958× | 0.966× | ✓ |  |
| C | increase_num_stages_for_software_pipelining_to_hide_l1tex_latency | 0.966× | 0.964× | ✓ |  |
| D | fix_autotune_static_grid_to_lambda_for_correct_xblock_scaling | 0.976× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | precompute_bn_sca... | Δ | increase_num_stag... | Δ | fix_autotune_stat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 11870 ns | 11810 ns | -60 ns (-0.5%) | 12290 ns | +420 ns (+3.5%) | 12320 ns | +450 ns (+3.8%) | 12320 ns | +450 ns (+3.8%) |
| SM Throughput (%) | 55.78 % | 33.81 % | -21.97 pp | 47.05 % | -8.73 pp | 44.71 % | -11.07 pp | 44.69 % | -11.09 pp |
| Memory Throughput (%) | 21.49 % | 21.43 % | -0.06 pp | 31.07 % | +9.58 pp | 30.88 % | +9.39 pp | 30.95 % | +9.46 pp |
| DRAM Throughput (%) | 12.96 % | 13.03 % | +0.07 pp | 12.50 % | -0.46 pp | 12.46 % | -0.50 pp | 12.47 % | -0.49 pp |
| Achieved Occupancy (%) | 60.92 % | 44.29 % | -16.63 pp | 43.85 % | -17.07 pp | 43.90 % | -17.02 pp | 42.69 % | -18.23 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 6.96 | 6.96 | +0.00 | 6.96 | +0.00 | 6.96 | +0.00 | 6.96 | +0.00 |
| No Eligible (%) | 31.01 % | 56.63 % | +25.62 pp | 38.83 % | +7.82 pp | 43.30 % | +12.29 pp | 41.80 % | +10.79 pp |
| IPC Active | 2.76 inst/cycle | 1.73 inst/cycle | -1.03 | 2.45 inst/cycle | -0.31 | 2.31 inst/cycle | -0.45 | 2.32 inst/cycle | -0.44 |

---

## Cumulative Progress

```
Baseline:  11870 ns  (1.000×)

Final best: 20380 ns  (0.5824×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1881600
@@ -19,23 +29,15 @@

     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    eps     = tl.full([1], 1e-05, tl.float32)
+    inv_std = tl.math.rsqrt(tmp3 + eps)
+    normed  = tmp2 * inv_std
+    tmp15   = normed * tmp12 + tmp14
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +47,7 @@

     arg_4 = inputs[4]
     xnumel = 1881600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
