# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 18690 ns — **Best latency**: 59900 ns (**0.3120×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_3_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 32
- **Duration**: 18690 ns
- **SM Throughput**: 67.23 % | **Memory**: 41.04 % | **DRAM**: 40.94 %
- **Achieved Occupancy**: 85.94 %
- **IPC Active**: 3.03 inst/cycle | **No Eligible**: 24.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_xblock_and_num_warps | 1.114× | 0.312× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 18690 ns | 59900 ns | +41210 ns (+220.5%) |
| SM Throughput (%) | 67.23 % | 76.16 % | +8.93 pp |
| Memory Throughput (%) | 41.04 % | 22.75 % | -18.29 pp |
| DRAM Throughput (%) | 40.94 % | 12.03 % | -28.91 pp |
| Achieved Occupancy (%) | 85.94 % | 66.45 % | -19.49 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 |
| Waves/SM | 7.42 | 19.80 | +12.38 |
| No Eligible (%) | 24.57 % | 21.04 % | -3.53 pp |
| IPC Active | 3.03 inst/cycle | 3.15 inst/cycle | +0.12 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps` (0.312×)

Adding triton.autotune with larger XBLOCK configs (256–1024) to a kernel where xnumel is exactly divisible by the original block size caused a 3.2x slowdown. The root cause was that the autotune wrapper changed the xmask from `tl.full([XBLOCK], True, tl.int1)` (a compile-time constant true mask) to `xindex < xnumel` (a runtime predicate), which forced all 9 masked loads and 2 stores to emit real predicate instructions. This caused executed instructions to balloon from 9.5M to 35.7M (3.7x increase), overwhelming any benefit from larger tiles. Waves per SM also increased from 7.42 to 19.8 (the autotune selected block_size=256, not a larger one), indicating the autotuner picked a suboptimal config that increased total work. When xnumel is exactly divisible by the original XBLOCK, the original `tl.full([XBLOCK], True, tl.int1)` mask is a zero-cost compile-time constant; replacing it with a runtime bounds check adds significant overhead across all memory operations.

> **Applicable when**: Never apply this pattern when xnumel is exactly divisible by the original XBLOCK and the kernel uses tl.full([XBLOCK], True, tl.int1) as its xmask — the constant mask is a critical optimization that must not be replaced with a runtime predicate.

> **Avoid when**: Do not add triton.autotune with variable XBLOCK when: (1) the original kernel uses a compile-time-constant True mask (tl.full([XBLOCK], True, tl.int1)) because xnumel is divisible by XBLOCK, AND (2) the kernel has many masked memory operations (9+ loads/stores) — converting the constant mask to a runtime predicate multiplies instruction count by 3-4x and causes severe regression regardless of tile size benefits.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_in_kernel_launcher | 1.113× | 0.312× | ✓ |  |
| B | reduce_register_pressure_via_maxnreg | 1.102× | 0.317× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_and_restore_static_mask | 0.986× | 1.005× | ✓ |  |
| D | restore_compile_time_constant_mask_remove_autotune | 1.004× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ | restore_compile_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18690 ns | 59810 ns | +41120 ns (+220.0%) | 58880 ns | +40190 ns (+215.0%) | 18590 ns | -100 ns (-0.5%) | 18620 ns | -70 ns (-0.4%) |
| SM Throughput (%) | 67.23 % | 76.07 % | +8.84 pp | 76.78 % | +9.55 pp | 67.27 % | +0.04 pp | 66.69 % | -0.54 pp |
| Memory Throughput (%) | 41.04 % | 22.72 % | -18.32 pp | 23.14 % | -17.90 pp | 41.26 % | +0.22 pp | 41.10 % | +0.06 pp |
| DRAM Throughput (%) | 40.94 % | 11.93 % | -29.01 pp | 11.93 % | -29.01 pp | 40.94 % | +0.00 pp | 40.87 % | -0.07 pp |
| Achieved Occupancy (%) | 85.94 % | 66.42 % | -19.52 pp | 89.89 % | +3.95 pp | 85.60 % | -0.34 pp | 85.93 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 7.42 | 19.80 | +12.38 | 14.85 | +7.43 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 24.57 % | 20.94 % | -3.63 pp | 20.51 % | -4.06 pp | 23.32 % | -1.25 pp | 23.24 % | -1.33 pp |
| IPC Active | 3.03 inst/cycle | 3.15 inst/cycle | +0.12 | 3.17 inst/cycle | +0.14 | 3.04 inst/cycle | +0.01 | 3.05 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan A: `fix_hardcoded_grid_in_kernel_launcher` (0.312×)

Changing the Triton kernel launcher grid from a static tuple `(triton.cdiv(xnumel, 128),)` to a lambda `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` produced no measurable improvement: duration changed from 59.9 µs to 59.81 µs (effectively identical), all profiling metrics remained unchanged, and the autotune still selected XBLOCK=256 with 19.8 waves/SM and 35.7M executed instructions. The root problem is not the grid expression but the structural regression introduced in iteration 1: the autotune wrapper replaced the compile-time-constant True mask (`tl.full([XBLOCK], True, tl.int1)`) with a runtime predicate (`xindex < xnumel`), causing a 3.7x instruction explosion that dominates all other effects. Fixing the launcher grid alone cannot recover performance when the kernel body itself is generating 3.7x more instructions due to predicated memory operations.

> **Applicable when**: Never: this fix alone is insufficient when the kernel body uses a runtime xmask predicate that inflates instruction count. The lambda grid is only meaningful if the kernel body can actually exploit larger XBLOCK values without predicate overhead.

> **Avoid when**: Do not apply a lambda grid fix in isolation when the kernel body already has a runtime bounds-check mask (`xindex < xnumel`) that causes instruction explosion. The grid expression is not the bottleneck when executed instruction count is 3.7x higher than the original kernel — the mask type in the kernel body must be fixed first (restore `tl.full([XBLOCK], True, tl.int1)` for divisible xnumel).

#### ❌ Plan B: `reduce_register_pressure_via_maxnreg` (0.317×)

Adding maxnreg=32 to triton.Config entries successfully lifted theoretical occupancy from 75% to 100% and achieved occupancy from 66.45% to 89.89% by moving Block Limit Registers from 6 to 8 blocks/SM. However, the kernel duration barely changed (59.9 µs → 58.88 µs) and the overall speedup was 0.32x (severe regression in wall-clock terms), because the CPIStall 'not selected' stall worsened: Warp Cycles Per Issued Instruction increased from 13.37 to 18.08 cycles, and the CPIStall not-selected fraction rose from 40.1% to 42.1%. The higher occupancy packed more warps onto each SM, but the kernel was already compute-bound (76% Compute SOL) with the ALU pipeline saturated — adding more warps only increased scheduler contention without hiding additional latency, and the net effect was neutral-to-negative on throughput. The regression in measured wall-clock time is likely dominated by other factors in the benchmark harness (e.g., the autotune overhead itself), since the kernel duration metrics are nearly identical.

> **Applicable when**: Never: when the kernel is already compute-bound (Compute SOL > 70%) and the dominant stall is CPIStall 'not selected', increasing occupancy via maxnreg does not help — the scheduler already has enough warps to hide latency and adding more only increases contention.

> **Avoid when**: Do not apply maxnreg to cap registers and boost occupancy when: (1) the kernel is already compute-bound (Compute SOL > 70%), (2) the dominant stall is CPIStall 'not selected' (warps eligible but not picked), AND (3) achieved occupancy is already above ~60%. In this regime, more warps increase scheduler contention (Warp Cycles Per Issued Instruction rises) without reducing the actual bottleneck. maxnreg is most beneficial when the kernel is memory-latency-bound and occupancy is the primary limiter for hiding that latency.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_epilogue_into_fma_pairs | 1.012× | 1.016× | ✓ |  |
| B | fuse_batchnorm_epilogue_arithmetic_into_fma | 1.000× | 1.000× | ✓ |  |
| C | restore_static_xblock_and_compile_time_constant_mask | 1.004× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_ep... | Δ | fuse_batchnorm_ep... | Δ | restore_static_xb... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 18690 ns | 18400 ns | -290 ns (-1.6%) | 18690 ns | +0 ns (+0.0%) | 18660 ns | -30 ns (-0.2%) |
| SM Throughput (%) | 67.23 % | 67.91 % | +0.68 pp | 66.88 % | -0.35 pp | 65.28 % | -1.95 pp |
| Memory Throughput (%) | 41.04 % | 41.73 % | +0.69 pp | 40.86 % | -0.18 pp | 40.93 % | -0.11 pp |
| DRAM Throughput (%) | 40.94 % | 41.50 % | +0.56 pp | 40.80 % | -0.14 pp | 40.93 % | -0.01 pp |
| Achieved Occupancy (%) | 85.94 % | 85.70 % | -0.24 pp | 85.68 % | -0.26 pp | 85.36 % | -0.58 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 24.57 % | 22.79 % | -1.78 pp | 23.65 % | -0.92 pp | 23.37 % | -1.20 pp |
| IPC Active | 3.03 inst/cycle | 3.09 inst/cycle | +0.06 | 3.04 inst/cycle | +0.01 | 3.04 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  18690 ns  (1.000×)

Final best: 59900 ns  (0.3120×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,21 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
+def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = tl.full([XBLOCK], True, tl.int1)
     x2 = ((xindex // 3584) % 56)
     x1 = ((xindex // 64) % 56)
     x0 = (xindex % 64)
     x6 = xindex // 3584
     x7 = xindex
-    tmp52 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp54 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp63 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp65 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp52 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp54 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp63 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp65 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = (-1) + 2*x2
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
@@ -31,46 +32,46 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x6), tmp10, other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x6), mask=tmp10 & xmask, other=float("-inf"))
     tmp12 = 2*x1
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x6), tmp16, other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x6), mask=tmp16 & xmask, other=float("-inf"))
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + 2*x1
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x6), tmp23, other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x6), mask=tmp23 & xmask, other=float("-inf"))
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = 2*x2
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x6), tmp30, other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x6), mask=tmp30 & xmask, other=float("-inf"))
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x6), tmp33, other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x6), mask=tmp33 & xmask, other=float("-inf"))
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x6), tmp36, other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x6), mask=tmp36 & xmask, other=float("-inf"))
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + 2*x2
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x6), tmp43, other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x6), mask=tmp43 & xmask, other=float("-inf"))
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x6), tmp46, other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x6), mask=tmp46 & xmask, other=float("-inf"))
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x6), tmp49, other=float("-inf"))
+    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x6), mask=tmp49 & xmask, other=float("-inf"))
     tmp51 = tl.maximum(tmp48, tmp50)
     tmp53 = tmp51 - tmp52
     tmp55 = tl.full([1], 1e-05, tl.float32)
@@ -85,10 +86,8 @@

     tmp66 = tmp64 + tmp65
     tmp67 = tl.full([1], 0, tl.int32)
     tmp68 = tl.maximum(tmp67, tmp66)
-    tl.store(out_ptr0 + (x7), tmp51, None)
-    tl.store(out_ptr1 + (x7), tmp68, None)
-
-
+    tl.store(out_ptr0 + (x7), tmp51, mask=xmask)
+    tl.store(out_ptr1 + (x7), tmp68, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
