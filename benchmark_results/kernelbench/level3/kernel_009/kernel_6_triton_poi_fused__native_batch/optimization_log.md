# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4740 ns — **Best latency**: 4740 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 22
- **Duration**: 4740 ns
- **SM Throughput**: 25.58 % | **Memory**: 13.98 % | **DRAM**: 13.98 %
- **Achieved Occupancy**: 62.99 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 46.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.913× | 0.862× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave | 0.913× | 0.862× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.960× | 0.566× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4740 ns | 5500 ns | +760 ns (+16.0%) | 5500 ns | +760 ns (+16.0%) | 8380 ns | +3640 ns (+76.8%) |
| SM Throughput (%) | 25.58 % | 48.84 % | +23.26 pp | 48.83 % | +23.25 pp | 62.25 % | +36.67 pp |
| Memory Throughput (%) | 13.98 % | 14.59 % | +0.61 pp | 14.56 % | +0.58 pp | 14.90 % | +0.92 pp |
| DRAM Throughput (%) | 13.98 % | 11.96 % | -2.02 pp | 11.99 % | -1.99 pp | 7.83 % | -6.15 pp |
| Achieved Occupancy (%) | 62.99 % | 75.19 % | +12.20 pp | 76.26 % | +13.27 pp | 81.04 % | +18.05 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.48 | 1.48 | +0.00 | 1.48 | +0.00 | 2.97 | +1.49 |
| No Eligible (%) | 46.57 % | 21.33 % | -25.24 pp | 21.13 % | -25.44 pp | 19.56 % | -27.01 pp |
| IPC Active | 1.93 inst/cycle | 3.12 inst/cycle | +1.19 | 2.99 inst/cycle | +1.06 | 3.25 inst/cycle | +1.32 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_to_eliminate_partial_wave` (0.862×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) to a pointwise kernel that NCU flagged for a 50% partial-wave tail caused a 16% regression. The autotuner selected a larger XBLOCK that increased registers per thread from 22 to 32, reducing Block Limit Registers from 21 to 16 blocks/SM. This cut the maximum schedulable blocks per SM, which combined with a 2.2x increase in executed instructions (from the xmask guard replacing tl.full([XBLOCK], True)) caused elapsed cycles to grow from 6973 to 8120. The L1/TEX hit rate also dropped from 70.2% to 58.3%, negating the expected cache-reuse benefit of larger tiles.

> **Applicable when**: Never apply this specific pattern. The strategy of increasing XBLOCK via autotune to eliminate partial waves is theoretically sound but failed here due to: (1) unnecessary xmask introduction inflating instruction count 2.2x, and (2) register pressure increase from 22 to 32 regs/thread reducing block-level parallelism.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) the original kernel uses tl.full([XBLOCK], True) masks (xnumel is exactly divisible by XBLOCK) and switching to xindex < xnumel masks would add unnecessary predication overhead; (2) the kernel already has low register usage near a block-limit-register boundary where larger configs would push registers per thread higher and reduce blocks/SM; (3) the kernel is short-duration (<10 us) where autotune overhead and instruction bloat outweigh partial-wave savings.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_unnecessary_xmask_and_fix_grid_lambda_for_autotune | 0.890× | 0.867× | ✓ |  |
| B | fix_autotune_grid_lambda_and_remove_unnecessary_xmask | 0.916× | 0.867× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.934× | 0.561× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_unnecessar... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4740 ns | 5470 ns | +730 ns (+15.4%) | 5470 ns | +730 ns (+15.4%) | 8450 ns | +3710 ns (+78.3%) |
| SM Throughput (%) | 25.58 % | 48.55 % | +22.97 pp | 49.32 % | +23.74 pp | 62.13 % | +36.55 pp |
| Memory Throughput (%) | 13.98 % | 14.45 % | +0.47 pp | 14.69 % | +0.71 pp | 14.88 % | +0.90 pp |
| DRAM Throughput (%) | 13.98 % | 12.06 % | -1.92 pp | 12.06 % | -1.92 pp | 7.80 % | -6.18 pp |
| Achieved Occupancy (%) | 62.99 % | 75.40 % | +12.41 pp | 74.91 % | +11.92 pp | 81.03 % | +18.04 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 1.48 | 1.48 | +0.00 | 1.48 | +0.00 | 2.97 | +1.49 |
| No Eligible (%) | 46.57 % | 21.21 % | -25.36 pp | 21.23 % | -25.34 pp | 17.69 % | -28.88 pp |
| IPC Active | 1.93 inst/cycle | 3.10 inst/cycle | +1.17 | 3.11 inst/cycle | +1.18 | 3.26 inst/cycle | +1.33 |


### Key Learnings

#### ❌ Plan C: `fix_autotune_grid_lambda_to_eliminate_instruction_explosion` (0.561×)

Attempting to fix the autotune grid mismatch by changing the call-site grid from a static tuple to a lambda (meta['XBLOCK']) while simultaneously replacing autotune configs (512/1024/2048 → 256/512/1024 with num_stages=1) still resulted in a regression: the grid remained at 3136 blocks and XBLOCK=256 was selected, doubling threads from 401K to 802K and increasing executed instructions from 2.13M to 4.10M (1.93x), raising duration from 5.5µs to 8.45µs. The profiling evidence shows the grid did NOT shrink despite the lambda fix — Block Size went from 128 to 256 but Grid Size stayed at 3136, indicating the lambda was either not applied correctly or the autotuner still selected XBLOCK=256 causing the same over-launch. The root cause of the instruction explosion (xmask predication overhead + register pressure from autotune) was not resolved by the grid lambda change alone.

> **Applicable when**: Never: this specific combination of autotune config changes + grid lambda fix on a kernel that already has xmask predication overhead and elevated register pressure (32 regs/thread) does not recover performance.

> **Avoid when**: Do not attempt to fix autotune grid mismatch regressions by only changing the grid lambda when: (1) the autotune configs still include small XBLOCK values (e.g., 256) that the autotuner may prefer due to register pressure; (2) the kernel already has xmask = xindex < xnumel predication adding instruction overhead; (3) registers per thread are already at 32, meaning larger XBLOCK configs hit Block Limit Registers = 8 blocks/SM. The correct fix requires reverting the entire autotune addition and restoring the original fixed XBLOCK=128 with tl.full([XBLOCK], True) masking.

---

## Cumulative Progress

```
Baseline:  4740 ns  (1.000×)

Final best: 4740 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 512}, num_warps=4),
+    triton.Config({'XBLOCK': 1024}, num_warps=8),
+    triton.Config({'XBLOCK': 2048}, num_warps=16),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 401408
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x2), None)
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x2), xmask)
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -33,9 +39,7 @@

     tmp17 = tmp15 + tmp16
     tmp18 = tl.full([1], 0, tl.int32)
     tmp19 = tl.maximum(tmp18, tmp17)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -46,7 +50,7 @@

     arg_5 = inputs[5]
     xnumel = 401408
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 401408, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 401408)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
