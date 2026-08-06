# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_hardswish_mean_native_group_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 186020 ns — **Best latency**: 186020 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_hardswish_mean_native_group_norm_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_027/kernel_1_triton_red_fused_convolution_h/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16384,1,1), Waves/SM: 7.76
- **Registers/thread**: 31
- **Duration**: 186020 ns
- **SM Throughput**: 74.95 % | **Memory**: 78.76 % | **DRAM**: 78.76 %
- **Achieved Occupancy**: 95.33 %
- **IPC Active**: 3.10 inst/cycle | **No Eligible**: 22.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | add_autotune_increase_xblock_for_coalesced_stores | 0.983× | 0.914× | ✓ |  |
| C | increase_xblock_for_coalesced_vectorized_access | 0.925× | 0.291× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_incr... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 186020 ns | 203520 ns | +17500 ns (+9.4%) | 638140 ns | +452120 ns (+243.0%) |
| SM Throughput (%) | 74.95 % | 85.35 % | +10.40 pp | 83.01 % | +8.06 pp |
| Memory Throughput (%) | 78.76 % | 71.96 % | -6.80 pp | 22.96 % | -55.80 pp |
| DRAM Throughput (%) | 78.76 % | 71.96 % | -6.80 pp | 22.96 % | -55.80 pp |
| Achieved Occupancy (%) | 95.33 % | 94.60 % | -0.73 pp | 30.69 % | -64.64 pp |
| Registers/Thread | 31.00 register/thread | 30.00 register/thread | -1 | 93.00 register/thread | +62 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 24.82 | +17.06 |
| No Eligible (%) | 22.32 % | 12.94 % | -9.38 pp | 16.35 % | -5.97 pp |
| IPC Active | 3.10 inst/cycle | 3.47 inst/cycle | +0.37 | 3.33 inst/cycle | +0.23 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_loop_invariant_rsqrt_and_reduce_nonfused_fp32_ops | 0.981× | 0.830× | ✓ |  |
| B | fuse_non_fused_fp32_instructions_via_fma_rewrite | 0.985× | 0.901× | ✓ |  |
| C | restore_xblock1_reduce_register_pressure_via_autotune_r0block_numwarps | 0.984× | 0.850× | ✓ |  |
| D | reduce_register_pressure_via_fma_and_expression_simplification | 0.931× | 0.292× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_loop_invari... | Δ | fuse_non_fused_fp... | Δ | restore_xblock1_r... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 186020 ns | 224030 ns | +38010 ns (+20.4%) | 206430 ns | +20410 ns (+11.0%) | 218780 ns | +32760 ns (+17.6%) | 636800 ns | +450780 ns (+242.3%) |
| SM Throughput (%) | 74.95 % | 79.76 % | +4.81 pp | 84.54 % | +9.59 pp | 83.86 % | +8.91 pp | 83.05 % | +8.10 pp |
| Memory Throughput (%) | 78.76 % | 65.40 % | -13.36 pp | 70.97 % | -7.79 pp | 66.99 % | -11.77 pp | 23.01 % | -55.75 pp |
| DRAM Throughput (%) | 78.76 % | 65.40 % | -13.36 pp | 70.97 % | -7.79 pp | 66.99 % | -11.77 pp | 23.01 % | -55.75 pp |
| Achieved Occupancy (%) | 95.33 % | 94.85 % | -0.48 pp | 94.84 % | -0.49 pp | 89.48 % | -5.85 pp | 36.74 % | -58.59 pp |
| Registers/Thread | 31.00 register/thread | 32.00 register/thread | +1 | 31.00 register/thread | +0 | 32.00 register/thread | +1 | 80.00 register/thread | +49 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 3.88 | -3.88 | 20.69 | +12.93 |
| No Eligible (%) | 22.32 % | 18.89 % | -3.43 pp | 14.05 % | -8.27 pp | 14.15 % | -8.17 pp | 16.33 % | -5.99 pp |
| IPC Active | 3.10 inst/cycle | 3.24 inst/cycle | +0.14 | 3.43 inst/cycle | +0.33 | 3.43 inst/cycle | +0.33 | 3.35 inst/cycle | +0.25 |


### Key Learnings

#### ❌ Plan A: `hoist_loop_invariant_rsqrt_and_reduce_nonfused_fp32_ops` (0.830×)

Explicitly hoisting a loop-invariant rsqrt (transcendental) call out of a Triton reduction loop caused a 17% regression (203us → 224us). Despite eliminating ~163K SFU instructions, executed instructions actually increased from 130.8M to 136.4M, branch instructions nearly doubled (802K → 1.52M), and eligible warps per scheduler dropped from 6.82 to 5.4 — suggesting the Triton compiler already performs LICM automatically and the manual hoist introduced additional overhead (likely from the restricted autotune config set removing higher-performing configs). The FPInstructions NCU rule flagging non-fused FP32 instructions does not reliably indicate that manual LICM will help when the compiler already handles it, and restricting autotune configs to XBLOCK=1-only simultaneously removed potentially better-performing tile configurations.

> **Applicable when**: Never — this specific combination of manual LICM hoist + restricted autotune configs caused regression. The Triton compiler likely already performs LICM on loop-invariant transcendental calls, making explicit hoisting redundant or harmful.

> **Avoid when**: Do not manually hoist loop-invariant transcendental calls (rsqrt, exp, log, etc.) in Triton kernels when: (1) the compiler is likely already performing LICM, (2) the change is bundled with restricting autotune configs (confounds the causal attribution), or (3) the FPInstructions NCU rule is the only signal — this rule flags non-fused vs fused FP32 counts but does not distinguish compiler-already-optimized LICM from genuinely redundant computation. Branch instruction count doubling post-change is a signal that the autotune config restriction (not the hoist) may be the primary regression cause.

#### ❌ Plan D: `reduce_register_pressure_via_fma_and_expression_simplification` (0.292×)

Collapsing a 14-intermediate sequential computation chain into 4 FMA-friendly expressions and hoisting loop-invariant rsqrt/scale computations outside the reduction loop did NOT reduce register pressure in Triton. Registers per thread increased from 30 to 80, collapsing theoretical occupancy from 100% to 37.5% (Block Limit Registers dropped from 16 to 6 blocks/SM). The root cause is that the autotune config changes (adding larger XBLOCK values) forced the compiler to keep a larger [XBLOCK, R0_BLOCK] accumulator live simultaneously, dominating over any savings from expression simplification. Total executed instructions grew 3.1x (130M → 411M) and duration tripled (203us → 637us), confirming the occupancy collapse was the primary regression driver.

> **Applicable when**: Never for this kernel configuration — expression simplification alone cannot overcome the register pressure introduced by XBLOCK tiling in Triton reduction kernels. Hoisting loop-invariant computations is generally sound but insufficient when the accumulator shape [XBLOCK, R0_BLOCK] dominates register usage.

> **Avoid when**: Kernel already achieves 100% theoretical occupancy with low register count (≤32 regs/thread). Any change that simultaneously modifies autotune configs to include larger XBLOCK values will inflate the [XBLOCK, R0_BLOCK] accumulator register footprint and collapse occupancy, regardless of how many intermediates are eliminated from the loop body. Do not combine expression simplification with XBLOCK expansion in the same change.

---

## Cumulative Progress

```
Baseline:  186020 ns  (1.000×)

Final best: 186020 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_hardswish_mean_native_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 16384
@@ -13,16 +28,16 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_base = tl.arange(0, R0_BLOCK)[None, :]
     rbase = r0_base
     x4 = xindex
     x0 = (xindex % 16)
-    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x4 // 4), None, eviction_policy='evict_last')
-    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x4 // 4), mask=xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x4 // 4), mask=xmask, eviction_policy='evict_last')
+    tmp21 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp23 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
     _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
@@ -30,7 +45,7 @@

         roffset = r0_offset
         rindex = r0_index
         r0_5 = r0_index
-        tmp0 = tl.load(in_ptr0 + (r0_5 + 10933*x4), r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (r0_5 + 10933*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.full([1, 1], 3.0, tl.float32)
         tmp4 = tmp2 + tmp3
@@ -57,9 +72,7 @@

     tmp28 = tl.full([1, 1], 10933.0, tl.float32)
     tmp29 = (tmp26 / tmp28)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x4), tmp29, None)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp29, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -72,7 +85,7 @@

     xnumel = 16384
     r0_numel = 10933
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_hardswish_mean_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 10933, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused_convolution_hardswish_mean_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 10933)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
