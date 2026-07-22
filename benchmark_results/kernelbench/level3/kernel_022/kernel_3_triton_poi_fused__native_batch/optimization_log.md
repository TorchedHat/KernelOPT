# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 407070 ns — **Best latency**: 407070 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_3_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (658560,1,1), Waves/SM: 311.82
- **Registers/thread**: 18
- **Duration**: 407070 ns
- **SM Throughput**: 42.61 % | **Memory**: 32.39 % | **DRAM**: 32.39 %
- **Achieved Occupancy**: 45.97 %
- **IPC Active**: 1.71 inst/cycle | **No Eligible**: 57.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_via_autotune_to_reduce_launch_overhead_and_hide_l1tex_latency | 0.999× | 0.334× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.998× | 0.334× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_better_occupancy | 0.998× | 0.886× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 0.998× | 0.326× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_v... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 407070 ns | 1220000 ns | +812930 ns (+199.7%) | 1220000 ns | +812930 ns (+199.7%) | 459230 ns | +52160 ns (+12.8%) | 1250000 ns | +842930 ns (+207.1%) |
| SM Throughput (%) | 42.61 % | 77.68 % | +35.07 pp | 77.84 % | +35.23 pp | 53.28 % | +10.67 pp | 73.42 % | +30.81 pp |
| Memory Throughput (%) | 32.39 % | 32.41 % | +0.02 pp | 32.48 % | +0.09 pp | 34.36 % | +1.97 pp | 31.55 % | -0.84 pp |
| DRAM Throughput (%) | 32.39 % | 10.81 % | -21.58 pp | 10.85 % | -21.54 pp | 28.74 % | -3.65 pp | 10.53 % | -21.86 pp |
| Achieved Occupancy (%) | 45.97 % | 71.54 % | +25.57 pp | 71.46 % | +25.49 pp | 36.60 % | -9.37 pp | 62.26 % | +16.29 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 34.00 register/thread | +16 |
| Waves/SM | 311.82 | 1247.27 | +935.45 | 1247.27 | +935.45 | 311.82 | +0.00 | 831.52 | +519.70 |
| No Eligible (%) | 57.23 % | 22.22 % | -35.01 pp | 22.14 % | -35.09 pp | 46.36 % | -10.87 pp | 26.48 % | -30.75 pp |
| IPC Active | 1.71 inst/cycle | 3.11 inst/cycle | +1.40 | 3.12 inst/cycle | +1.41 | 2.17 inst/cycle | +0.46 | 2.94 inst/cycle | +1.23 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_via_autotune_to_reduce_launch_overhead_and_hide_l1tex_latency` (0.334×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that already had low register pressure (18 regs/thread) caused the compiler to select a config (XBLOCK=512, num_warps=8) that doubled register usage to 32/thread and increased executed instructions by 5.4x (137M → 744M). Despite improving compute SM throughput (43% → 78%) and occupancy (46% → 72%), the massive instruction count increase drove duration from 407µs to 1.22ms — a 3x regression. The autotune overhead itself (multiple trial runs) also contributes to wall-time regression in benchmarks that measure total time including autotuning. When the kernel body is already efficient and the bottleneck is scheduling overhead from too many small CTAs, adding autotune with larger blocks can backfire if the compiler generates more instructions per element at larger tile sizes due to register pressure or predication overhead.

> **Applicable when**: Never apply this specific pattern. The general idea of increasing XBLOCK to hide L1TEX latency is sound, but using @triton.autotune to do so on a kernel with very high waves_per_sm (>300) and low register count (<20 regs/thread) can cause the compiler to generate significantly more instructions at larger tile sizes.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs when: (1) the kernel already has low register pressure (≤20 regs/thread) and the compiler may increase register usage at larger tiles; (2) the benchmark measures total wall time including autotune trial runs; (3) the grid is already large (>600K CTAs) and waves_per_sm is already high (>300) — the kernel is not actually bottlenecked by insufficient parallelism. Instead, consider manually setting a single larger XBLOCK without autotune overhead.

#### ❌ Plan B: `autotune_xblock_and_num_warps_for_pointwise_kernel` (0.334×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise kernel that was already running 311 waves/SM caused a severe 3x regression. The autotuner selected XBLOCK=512 with 32 registers/thread (up from 18), which increased total instruction count by 5.4x (137M → 744M) and waves/SM by 4x (312 → 1247), overwhelming the GPU with redundant work. Although compute throughput improved (43% → 78%) and L1TEX stalls decreased, the massive increase in total executed instructions and waves dominated, making the kernel 3x slower. When a kernel already has very high wave count (>300 waves/SM) and low register pressure, increasing XBLOCK does not reduce total work — it multiplies it if the grid is not correspondingly reduced.

> **Applicable when**: Never — this specific pattern (fixed grid not updated to match autotuned XBLOCK) is always wrong. As a general negative: do not apply autotune XBLOCK expansion when the grid launch count is not correspondingly reduced (i.e., grid must be a lambda using meta['XBLOCK']).

> **Avoid when**: Any time the grid is a static tuple computed with the original XBLOCK rather than a meta-lambda. Also avoid when the kernel already has >200 waves/SM and register pressure is low — the bottleneck may not be addressable by XBLOCK scaling alone, and the autotuner overhead plus increased instruction count can dominate.

#### ❌ Plan D: `autotune_xblock_increase_tile_size` (0.326×)

Adding @triton.autotune with larger XBLOCK configs to a pointwise kernel while keeping the launch grid hardcoded to cdiv(xnumel, 128) caused the autotuner to select XBLOCK=256, doubling the number of elements processed per CTA while the grid size remained unchanged. This resulted in a 3.8x increase in executed instructions (191M → 724M) and a 2.7x increase in kernel duration (459µs → 1250µs). Although eligible warps per scheduler improved (0.8 → 4.83) and compute throughput rose (53% → 73%), the massive instruction redundancy dominated, causing a severe regression. The root cause is that the grid must be a lambda `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` whenever XBLOCK is autotuned; a static grid hardcoded to the original XBLOCK causes every element to be processed multiple times proportional to the XBLOCK ratio.

> **Applicable when**: Never apply this pattern as implemented. The general lesson: whenever @triton.autotune is used with variable XBLOCK, the grid MUST be a lambda using meta['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK configs without simultaneously changing the grid to `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. Also avoid when the kernel already has >300 waves/SM — the bottleneck is not insufficient parallelism, and larger XBLOCK increases register pressure (18→34 regs/thread observed), capping theoretical occupancy at 75% instead of 100%.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_for_autotune_xblock_adaptation | 0.998× | 0.890× | ✓ |  |
| B | remove_broken_autotune_restore_correct_fixed_config | 1.000× | 1.000× | ✓ |  |
| C | replace_non_fused_fp32_chain_with_fma_friendly_arithmetic | 0.998× | 0.331× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_for_autotuned_xblock | 0.998× | 0.326× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | remove_broken_aut... | Δ | replace_non_fused... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 407070 ns | 457310 ns | +50240 ns (+12.3%) | 406940 ns | -130 ns (-0.0%) | 1230000 ns | +822930 ns (+202.2%) | 1250000 ns | +842930 ns (+207.1%) |
| SM Throughput (%) | 42.61 % | 53.76 % | +11.15 pp | 42.60 % | -0.01 pp | 77.05 % | +34.44 pp | 73.44 % | +30.83 pp |
| Memory Throughput (%) | 32.39 % | 34.66 % | +2.27 pp | 32.40 % | +0.01 pp | 32.23 % | -0.16 pp | 31.57 % | -0.82 pp |
| DRAM Throughput (%) | 32.39 % | 28.85 % | -3.54 pp | 32.40 % | +0.01 pp | 10.75 % | -21.64 pp | 10.55 % | -21.84 pp |
| Achieved Occupancy (%) | 45.97 % | 36.89 % | -9.08 pp | 45.94 % | -0.03 pp | 76.41 % | +30.44 pp | 62.40 % | +16.43 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 34.00 register/thread | +16 |
| Waves/SM | 311.82 | 311.82 | +0.00 | 311.82 | +0.00 | 1247.27 | +935.45 | 831.52 | +519.70 |
| No Eligible (%) | 57.23 % | 45.91 % | -11.32 pp | 57.18 % | -0.05 pp | 22.67 % | -34.56 pp | 26.46 % | -30.77 pp |
| IPC Active | 1.71 inst/cycle | 2.15 inst/cycle | +0.44 | 1.71 inst/cycle | +0.00 | 3.08 inst/cycle | +1.37 | 2.94 inst/cycle | +1.23 |


### Key Learnings

#### ❌ Plan C: `replace_non_fused_fp32_chain_with_fma_friendly_arithmetic` (0.331×)

Applying tl.math.fma() to restructure a batch-norm + hardtanh arithmetic chain while simultaneously expanding XBLOCK via @triton.autotune caused a severe 3x regression (459µs → 1230µs). The root cause is not the FMA restructuring itself but the autotune XBLOCK expansion: the autotuner selected XBLOCK=512 with a fixed grid of 658560 CTAs (not scaled down), causing 3.9x more executed instructions (191M → 744M) and 4x more waves/SM (312 → 1247). Although compute throughput improved (53% → 77%) and occupancy improved (37% → 76%), the massive instruction count increase dominated. The FPInstructions NCU rule suggesting FMA fusion is misleading when the optimization plan also changes the autotune configuration — the instruction explosion from the grid/XBLOCK mismatch dwarfs any FMA savings.

> **Applicable when**: Never apply this combined pattern. tl.math.fma() restructuring alone may be neutral or mildly positive, but cannot be evaluated when entangled with an autotune XBLOCK expansion that uses a fixed grid.

> **Avoid when**: Do not combine FMA arithmetic restructuring with @triton.autotune XBLOCK expansion in the same change. Do not add autotune configs with larger XBLOCK when the grid is a static tuple (not a lambda using meta['XBLOCK']). Do not apply when the kernel already has >300 waves/SM — the bottleneck is not insufficient parallelism. The FPInstructions NCU rule alone is insufficient justification when the kernel is already memory/latency bound at the original block size.

---

## Cumulative Progress

```
Baseline:  407070 ns  (1.000×)

Final best: 407070 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+], key=['xnumel'], restore_value=['in_out_ptr0'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 84295680
@@ -31,7 +38,6 @@

     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -40,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 84295680
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 84295680, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 84295680)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
