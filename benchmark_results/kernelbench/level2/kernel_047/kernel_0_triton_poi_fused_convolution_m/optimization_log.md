# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_mish_tanh_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 639460 ns — **Best latency**: 639460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_mish_tanh_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_047/kernel_0_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (922560,1,1), Waves/SM: 436.82
- **Registers/thread**: 16
- **Duration**: 639460 ns
- **SM Throughput**: 82.40 % | **Memory**: 29.23 % | **DRAM**: 29.23 %
- **Achieved Occupancy**: 80.84 %
- **IPC Active**: 3.31 inst/cycle | **No Eligible**: 17.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_persistent_tile_scheduling | 0.998× | 0.107× | ✓ |  |
| B | increase_xblock_with_autotune_for_compute_bound_pointwise | 0.669× | 0.111× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_transcendental_latency_hiding | 0.998× | 0.405× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | 0.384× | 0.405× | ✓ |  |
| E | add_triton_autotune | 0.999× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_p... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 639460 ns | 6000000 ns | +5360540 ns (+838.3%) | 5780000 ns | +5140540 ns (+803.9%) | 1580000 ns | +940540 ns (+147.1%) | 1580000 ns | +940540 ns (+147.1%) | 640290 ns | +830 ns (+0.1%) |
| SM Throughput (%) | 82.40 % | 92.04 % | +9.64 pp | 92.72 % | +10.32 pp | 91.39 % | +8.99 pp | 91.43 % | +9.03 pp | 83.59 % | +1.19 pp |
| Memory Throughput (%) | 29.23 % | 8.06 % | -21.17 pp | 8.33 % | -20.90 pp | 11.85 % | -17.38 pp | 11.82 % | -17.41 pp | 29.18 % | -0.05 pp |
| DRAM Throughput (%) | 29.23 % | 3.11 % | -26.12 pp | 3.24 % | -25.99 pp | 11.85 % | -17.38 pp | 11.82 % | -17.41 pp | 29.18 % | -0.05 pp |
| Achieved Occupancy (%) | 80.84 % | 83.35 % | +2.51 pp | 90.53 % | +9.69 pp | 89.11 % | +8.27 pp | 88.62 % | +7.78 pp | 80.96 % | +0.12 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 32.00 register/thread | +16 | 28.00 register/thread | +12 | 28.00 register/thread | +12 | 16.00 register/thread | +0 |
| Waves/SM | 436.82 | 1747.27 | +1310.45 | 873.64 | +436.82 | 436.82 | +0.00 | 436.82 | +0.00 | 436.82 | +0.00 |
| No Eligible (%) | 17.27 % | 7.91 % | -9.36 pp | 7.23 % | -10.04 pp | 8.42 % | -8.85 pp | 8.45 % | -8.82 pp | 16.12 % | -1.15 pp |
| IPC Active | 3.31 inst/cycle | 3.68 inst/cycle | +0.37 | 3.71 inst/cycle | +0.40 | 3.66 inst/cycle | +0.35 | 3.66 inst/cycle | +0.35 | 3.36 inst/cycle | +0.05 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_persistent_tile_scheduling` (0.107×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a pointwise kernel that already has 436 waves/SM caused a 9.4x regression. The grid lambda was not updated to use meta['XBLOCK'], so the grid remained fixed at cdiv(N, 128) while autotune selected XBLOCK=512, causing each CTA to process 4x more elements than the grid allocated — resulting in 10x more total instructions (417M → 4375M) and duration jumping from 639 µs to 6 ms. The register count also increased from 16 to 28 per thread, reducing occupancy headroom. The root cause was an implementation flaw: the grid lambda must be updated to lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),) when using autotune with variable XBLOCK.

> **Applicable when**: Never apply this broken pattern. When using @triton.autotune with variable XBLOCK, always use a dynamic grid lambda.

> **Avoid when**: Do not use a static grid = (cdiv(N, fixed_block),) when @triton.autotune is varying XBLOCK. The grid must be a callable lambda meta: (cdiv(N, meta['XBLOCK']),) so it scales with the autotuned tile size.

#### ❌ Plan B: `increase_xblock_with_autotune_for_compute_bound_pointwise` (0.111×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise Triton kernel caused a 9x slowdown (639µs → 5780µs) despite improving achieved occupancy (80.8% → 90.5%) and SM throughput (82.4% → 92.7%). The root cause is that the selected config (XBLOCK=256) doubled registers per thread (16→32) and caused total executed instructions to balloon ~10x (417M → 4241M), indicating the compiler generated far more instructions per element at the larger tile size — likely due to predication overhead from the incorrectly introduced xmask (the diff changed `tl.full([XBLOCK], True, tl.int1)` to `xindex < xnumel`, adding unnecessary masking to all loads/stores despite xnumel being exactly divisible). Additionally, the grid lambda was not updated to use `meta['XBLOCK']`, so the grid remained fixed at cdiv(xnumel, 128) regardless of the autotuned XBLOCK, causing each CTA to process far more elements than its XBLOCK tile — a correctness and performance bug. When autotune is applied but the grid is not updated to use the autotuned tile size, the kernel executes incorrectly and with massive overhead.

> **Applicable when**: Never apply this exact pattern. The strategy of increasing XBLOCK via autotune is sound in principle, but requires: (1) grid lambda updated to use meta['XBLOCK'], (2) masking only added if xnumel is NOT exactly divisible by all candidate XBLOCKs.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs without simultaneously updating the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Do not introduce xmask = xindex < xnumel when xnumel is exactly divisible by all candidate XBLOCK values — this adds unnecessary predication that inflates instruction count ~10x. Do not apply when registers per thread will double (16→32) due to larger tile, as this halves max blocks per SM.

#### ❌ Plan C: `autotune_xblock_and_num_warps_for_transcendental_latency_hiding` (0.405×)

Adding @triton.autotune with larger XBLOCK configs caused the Triton compiler to allocate significantly more registers per thread (16 → 28) and dramatically increased total executed instructions (~417M → ~1140M, a 2.7× increase), resulting in a 2.5× longer kernel duration despite improved scheduler eligibility. The autotune decorator itself, combined with the introduction of a non-trivial xmask (xindex < xnumel) where the original kernel used tl.full([XBLOCK], True), forced the compiler to emit additional predication and masking instructions across all 118M elements. When the original kernel already uses tl.full([XBLOCK], True) as its mask (i.e., no masking needed), replacing it with a real bounds-check mask adds unnecessary instruction overhead at scale.

> **Applicable when**: Never applicable in this form. The autotune approach with larger XBLOCK is theoretically sound for latency hiding, but the implementation introduced two compounding bugs: (1) the grid was not updated to use meta['XBLOCK'], so the grid remained fixed at cdiv(xnumel, 128) regardless of the autotuned XBLOCK, causing massive work duplication; (2) replacing tl.full([XBLOCK], True) with xindex < xnumel added unnecessary masking overhead when xnumel is evenly divisible by XBLOCK.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK when: (1) the launch grid is hardcoded and not updated to use meta['XBLOCK'] — this causes the grid to be computed for the old XBLOCK while the kernel uses a larger XBLOCK, leading to massive redundant work; (2) the original kernel uses tl.full([XBLOCK], True) as mask (meaning xnumel is exactly divisible by XBLOCK) — introducing a real bounds-check mask adds unnecessary predication instructions at scale; (3) the kernel is already compute-bound at >80% SM throughput with low register pressure (16 regs/thread) — autotune overhead from recompilation and the grid mismatch bug will dominate.

#### ❌ Plan D: `increase_xblock_for_vectorized_loads_and_reduced_launch_overhead` (0.405×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a Triton pointwise kernel that was already compute-bound (Compute SOL 82%) caused a 2.47x regression. The executed instruction count ballooned from 417M to 1142M (~2.7x), while the grid size remained unchanged at 922K blocks (the lambda grid was not wired up — the fixed grid of cdiv(xnumel, 128) was kept in kernel_function, so autotune selected a larger XBLOCK but the grid was not reduced). The larger XBLOCK combined with the restored xmask boundary check introduced additional predicated instructions and register pressure (16→28 registers/thread), increasing compute time without reducing memory traffic or launch overhead.

> **Applicable when**: Never apply this exact pattern — it is an implementation flaw. The general strategy of increasing XBLOCK with autotune is applicable when: kernel is memory-bandwidth-bound (Memory SOL > 60%), registers/thread are low (<20), and grid size is very large (>100K blocks).

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs without simultaneously converting the launch grid to a lambda (lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)). If the grid remains fixed at the original small XBLOCK size, each thread block will process more elements than the grid covers, causing redundant work or out-of-bounds access. Also avoid on already compute-bound kernels (Compute SOL > 80%) where increasing tile size adds instruction overhead without reducing memory bottlenecks.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_xblock_exploration | 0.998× | 0.986× | ✓ |  |
| B | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.999× | 0.989× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.384× | 0.405× | ✓ |  |
| D | fix_broken_autotune_grid_and_restore_optimal_xblock128 | 1.000× | 0.995× | ✓ |  |
| E | add_triton_autotune | 0.999× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_broken_autotu... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 639460 ns | 648420 ns | +8960 ns (+1.4%) | 646850 ns | +7390 ns (+1.2%) | 1580000 ns | +940540 ns (+147.1%) | 642940 ns | +3480 ns (+0.5%) | 641250 ns | +1790 ns (+0.3%) |
| SM Throughput (%) | 82.40 % | 83.75 % | +1.35 pp | 83.71 % | +1.31 pp | 91.41 % | +9.01 pp | 82.45 % | +0.05 pp | 83.61 % | +1.21 pp |
| Memory Throughput (%) | 29.23 % | 28.81 % | -0.42 pp | 28.88 % | -0.35 pp | 11.81 % | -17.42 pp | 29.06 % | -0.17 pp | 29.13 % | -0.10 pp |
| DRAM Throughput (%) | 29.23 % | 28.81 % | -0.42 pp | 28.88 % | -0.35 pp | 11.81 % | -17.42 pp | 29.06 % | -0.17 pp | 29.13 % | -0.10 pp |
| Achieved Occupancy (%) | 80.84 % | 81.13 % | +0.29 pp | 81.09 % | +0.25 pp | 88.40 % | +7.56 pp | 80.91 % | +0.07 pp | 80.76 % | -0.08 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 28.00 register/thread | +12 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 436.82 | 436.82 | +0.00 | 436.82 | +0.00 | 436.82 | +0.00 | 436.82 | +0.00 | 436.82 | +0.00 |
| No Eligible (%) | 17.27 % | 15.95 % | -1.32 pp | 15.98 % | -1.29 pp | 8.43 % | -8.84 pp | 17.18 % | -0.09 pp | 16.09 % | -1.18 pp |
| IPC Active | 3.31 inst/cycle | 3.36 inst/cycle | +0.05 | 3.36 inst/cycle | +0.05 | 3.66 inst/cycle | +0.35 | 3.31 inst/cycle | +0.00 | 3.35 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  639460 ns  (1.000×)

Final best: 639460 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,16 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+], key=['xnumel'], restore_value=['in_out_ptr0'])
 @triton.jit
 def triton_poi_fused_convolution_mish_tanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 118087680
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x3 = xindex
     x1 = ((xindex // 115320) % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x3), None)
-    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
+    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 20.0, tl.float32)
     tmp4 = tmp2 > tmp3
@@ -24,16 +31,14 @@

     tmp8 = libdevice.tanh(tmp7)
     tmp9 = tmp2 * tmp8
     tmp10 = libdevice.tanh(tmp9)
-    tl.store(in_out_ptr0 + (x3), tmp10, None)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 118087680
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_convolution_mish_tanh_0[grid](arg_0, arg_1, 118087680, XBLOCK=128)
+    triton_poi_fused_convolution_mish_tanh_0[grid](arg_0, arg_1, 118087680)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
