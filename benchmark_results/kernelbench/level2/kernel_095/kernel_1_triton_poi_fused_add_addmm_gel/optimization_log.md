# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 44320 ns — **Best latency**: 45120 ns (**0.9823×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_095/kernel_1_triton_poi_fused_add_addmm_gel/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 16
- **Duration**: 44320 ns
- **SM Throughput**: 77.59 % | **Memory**: 19.89 % | **DRAM**: 18.62 %
- **Achieved Occupancy**: 77.73 %
- **IPC Active**: 3.27 inst/cycle | **No Eligible**: 17.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.286× | 0.218× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_compute_bound_pointwise | 1.001× | 0.988× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | 0.535× | 0.416× | ✓ |  |
| D | autotune_xblock_for_fused_fp32_and_occupancy | 1.001× | 0.112× | ✓ |  |
| E | add_triton_autotune | 1.002× | 0.982× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_f... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44320 ns | 203100 ns | +158780 ns (+358.3%) | 44860 ns | +540 ns (+1.2%) | 106400 ns | +62080 ns (+140.1%) | 395230 ns | +350910 ns (+791.8%) | 45120 ns | +800 ns (+1.8%) |
| SM Throughput (%) | 77.59 % | 89.86 % | +12.27 pp | 79.95 % | +2.36 pp | 89.72 % | +12.13 pp | 91.31 % | +13.72 pp | 80.08 % | +2.49 pp |
| Memory Throughput (%) | 19.89 % | 7.49 % | -12.40 pp | 19.66 % | -0.23 pp | 9.57 % | -10.32 pp | 6.50 % | -13.39 pp | 19.56 % | -0.33 pp |
| DRAM Throughput (%) | 18.62 % | 4.08 % | -14.54 pp | 18.42 % | -0.20 pp | 7.83 % | -10.79 pp | 2.10 % | -16.52 pp | 18.68 % | +0.06 pp |
| Achieved Occupancy (%) | 77.73 % | 94.37 % | +16.64 pp | 79.38 % | +1.65 pp | 91.75 % | +14.02 pp | 95.14 % | +17.41 pp | 79.23 % | +1.50 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 16.00 register/thread | +0 | 28.00 register/thread | +12 | 32.00 register/thread | +16 | 16.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 62.06 | +31.03 | 31.03 | +0.00 |
| No Eligible (%) | 17.68 % | 9.07 % | -8.61 pp | 16.15 % | -1.53 pp | 8.45 % | -9.23 pp | 8.14 % | -9.54 pp | 15.57 % | -2.11 pp |
| IPC Active | 3.27 inst/cycle | 3.64 inst/cycle | +0.37 | 3.36 inst/cycle | +0.09 | 3.66 inst/cycle | +0.39 | 3.67 inst/cycle | +0.40 | 3.36 inst/cycle | +0.09 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.218×)

Adding @triton.autotune with larger XBLOCK configs caused a 4.6x slowdown (44 µs → 203 µs) despite occupancy improving from 77.7% to 94.4%. The root cause is that the launch grid was left fixed at cdiv(xnumel, 128) instead of being updated to a lambda using meta['XBLOCK'], so the autotuner selected a larger XBLOCK but the grid remained sized for XBLOCK=128 — causing each block to process far more elements than intended and inflating executed instructions from 26.7M to 144.2M (~5.4x). Additionally, registers per thread doubled from 16 to 32, halving the theoretical blocks-per-SM. When using autotune with variable XBLOCK, the launch grid must be a lambda that reads meta['XBLOCK'], not a static expression computed from the original tile size.

> **Applicable when**: Never apply this pattern (fixed grid + autotune XBLOCK). The lambda grid form is always required when autotuning XBLOCK.

> **Avoid when**: Do not use a static grid expression when adding @triton.autotune over XBLOCK. The static grid will be computed from the original XBLOCK value and will not adapt to the autotuner's chosen config, causing each block to process XBLOCK_new/XBLOCK_old times more work than intended.

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_reduced_launch_overhead` (0.416×)

Adding @triton.autotune with larger XBLOCK values (512–4096) to a compute-bound pointwise kernel caused a 2.4x slowdown despite improving achieved occupancy from 77.7% to 91.8%. The total executed instruction count ballooned from 26.7M to 75.1M (2.8x increase), and registers per thread jumped from 16 to 28, indicating the compiler unrolled more work per thread and increased register pressure without a compensating reduction in grid size (grid remained 65536 because the autotune wrapper did not update the grid lambda). The memory throughput actually dropped from 19.9% to 9.6%, confirming the kernel became more compute-bound with redundant work rather than more efficient. The implementation flaw was that the static grid=(cdiv(xnumel, 128),) was not updated to use meta['XBLOCK'], so the grid stayed at 65536 blocks regardless of which XBLOCK config was selected, causing each block to process far fewer elements than XBLOCK while still paying the full unrolled instruction cost.

> **Applicable when**: Never apply this specific pattern (static grid with autotune). The general XBLOCK increase strategy is applicable when: memory throughput is the bottleneck (Memory SOL >> Compute SOL), grid size is very large (>32K blocks), and registers per thread are low (<20), leaving headroom for larger tiles.

> **Avoid when**: Do not use @triton.autotune with a static grid that hardcodes the original XBLOCK in cdiv — the grid will not shrink with larger XBLOCK values, causing each CTA to process fewer elements than XBLOCK while the compiler still unrolls for the full XBLOCK, exploding instruction count. Also avoid increasing XBLOCK when the kernel is already compute-bound (Compute SOL > 75%) rather than memory-bound.

#### ❌ Plan D: `autotune_xblock_for_fused_fp32_and_occupancy` (0.112×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a Triton pointwise kernel caused a ~9x regression (44 µs → 395 µs). The autotuner selected XBLOCK=256 with 32 registers/thread (doubled from 16), and executed instructions exploded from 26.7M to 285.3M — a 10.7x increase — indicating the larger block size triggered excessive loop unrolling or redundant computation rather than FMA fusion. The grid was not updated to use META['XBLOCK'], so the grid remained fixed at cdiv(xnumel, 128) while the kernel processed XBLOCK=256 elements per CTA, causing each element to be processed multiple times (2x overlap), which explains the instruction count explosion.

> **Applicable when**: Never apply this specific pattern (fixed grid + autotune without META grid lambda). The correct autotune pattern with a META grid lambda may be applicable when a Triton pointwise kernel is compute-bound with low register pressure (<20 registers/thread) and high non-fused FP32 instruction ratio.

> **Avoid when**: Do NOT add @triton.autotune without simultaneously updating the grid to use `lambda META: (triton.cdiv(xnumel, META['XBLOCK']),)`. If the grid remains hardcoded to the original XBLOCK while the autotuner selects a larger XBLOCK, each element will be processed multiple times, causing instruction count explosion and severe regression. Also avoid when register count is already near the SM limit, as larger XBLOCK may double register usage and reduce occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_xblock_tuning | 0.998× | 0.992× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_correct_xblock_exploration | 1.000× | 0.985× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.537× | 0.418× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.287× | 0.218× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44320 ns | 44670 ns | +350 ns (+0.8%) | 44990 ns | +670 ns (+1.5%) | 106140 ns | +61820 ns (+139.5%) | 203140 ns | +158820 ns (+358.3%) |
| SM Throughput (%) | 77.59 % | 79.85 % | +2.26 pp | 80.08 % | +2.49 pp | 89.61 % | +12.02 pp | 89.91 % | +12.32 pp |
| Memory Throughput (%) | 19.89 % | 19.77 % | -0.12 pp | 19.62 % | -0.27 pp | 9.57 % | -10.32 pp | 7.49 % | -12.40 pp |
| DRAM Throughput (%) | 18.62 % | 18.63 % | +0.01 pp | 18.44 % | -0.18 pp | 7.79 % | -10.83 pp | 4.07 % | -14.55 pp |
| Achieved Occupancy (%) | 77.73 % | 79.76 % | +2.03 pp | 79.25 % | +1.52 pp | 91.63 % | +13.90 pp | 94.27 % | +16.54 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 28.00 register/thread | +12 | 32.00 register/thread | +16 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 17.68 % | 15.74 % | -1.94 pp | 15.99 % | -1.69 pp | 8.47 % | -9.21 pp | 9.04 % | -8.64 pp |
| IPC Active | 3.27 inst/cycle | 3.36 inst/cycle | +0.09 | 3.35 inst/cycle | +0.08 | 3.66 inst/cycle | +0.39 | 3.64 inst/cycle | +0.37 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_existing_autotune_to_work_correctly | 1.001× | 0.992× | ✓ |  |
| B | increase_num_warps_to_hide_transcendental_latency | 0.996× | 0.993× | ✓ |  |
| C | remove_broken_autotune_to_eliminate_instruction_explosion | 1.001× | 0.981× | ✓ |  |
| D | fix_grid_lambda_to_enable_correct_autotune_exploration | 0.287× | 0.218× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | increase_num_warp... | Δ | remove_broken_aut... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44320 ns | 44670 ns | +350 ns (+0.8%) | 44640 ns | +320 ns (+0.7%) | 45180 ns | +860 ns (+1.9%) | 203520 ns | +159200 ns (+359.2%) |
| SM Throughput (%) | 77.59 % | 80.22 % | +2.63 pp | 79.87 % | +2.28 pp | 80.06 % | +2.47 pp | 89.94 % | +12.35 pp |
| Memory Throughput (%) | 19.89 % | 19.76 % | -0.13 pp | 19.79 % | -0.10 pp | 19.50 % | -0.39 pp | 7.49 % | -12.40 pp |
| DRAM Throughput (%) | 18.62 % | 18.67 % | +0.05 pp | 18.55 % | -0.07 pp | 18.42 % | -0.20 pp | 4.10 % | -14.52 pp |
| Achieved Occupancy (%) | 77.73 % | 79.69 % | +1.96 pp | 79.61 % | +1.88 pp | 79.57 % | +1.84 pp | 94.40 % | +16.67 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 32.00 register/thread | +16 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 17.68 % | 16.00 % | -1.68 pp | 15.54 % | -2.14 pp | 16.32 % | -1.36 pp | 9.15 % | -8.53 pp |
| IPC Active | 3.27 inst/cycle | 3.37 inst/cycle | +0.10 | 3.37 inst/cycle | +0.10 | 3.37 inst/cycle | +0.10 | 3.63 inst/cycle | +0.36 |

---

## Cumulative Progress

```
Baseline:  44320 ns  (1.000×)

Final best: 45120 ns  (0.9823×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 8192)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
     tmp5 = tl.sigmoid(tmp4)
@@ -32,9 +46,7 @@

     tmp16 = tl.full([1], -1.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.minimum(tmp17, tmp13)
-    tl.store(in_out_ptr0 + (x2), tmp18, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -42,7 +54,7 @@

     arg_2 = inputs[2]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1[grid](arg_0, arg_1, arg_2, 8388608, XBLOCK=128)
+    triton_poi_fused_add_addmm_gelu_hardtanh_mul_sigmoid_tanh_1[grid](arg_0, arg_1, arg_2, 8388608)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
