# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11810 ns — **Best latency**: 11780 ns (**1.0025×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_for_occupancy_and_l1_reuse`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (14700,1,1), Waves/SM: 6.96
- **Registers/thread**: 18
- **Duration**: 11810 ns
- **SM Throughput**: 55.37 % | **Memory**: 25.58 % | **DRAM**: 13.02 %
- **Achieved Occupancy**: 61.65 %
- **IPC Active**: 2.76 inst/cycle | **No Eligible**: 30.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.998× | 0.223× | ✓ |  |
| B | autotune_xblock_for_occupancy_and_l1_reuse | 1.014× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11810 ns | 52930 ns | +41120 ns (+348.2%) | 11780 ns | -30 ns (-0.3%) |
| SM Throughput (%) | 55.37 % | 83.95 % | +28.58 pp | 55.42 % | +0.05 pp |
| Memory Throughput (%) | 25.58 % | 17.62 % | -7.96 pp | 25.57 % | -0.01 pp |
| DRAM Throughput (%) | 13.02 % | 2.90 % | -10.12 pp | 13.02 % | +0.00 pp |
| Achieved Occupancy (%) | 61.65 % | 84.85 % | +23.20 pp | 61.28 % | -0.37 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 6.96 | 27.84 | +20.88 | 6.96 | +0.00 |
| No Eligible (%) | 30.79 % | 12.30 % | -18.49 pp | 30.75 % | -0.04 pp |
| IPC Active | 2.76 inst/cycle | 3.50 inst/cycle | +0.74 | 2.76 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.223×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused a 4.5x slowdown despite improving achieved occupancy from 61.6% to 84.9%. The autotune selected XBLOCK=512, which increased registers per thread from 18 to 32 (Block Limit Registers dropped from 21 to 4 blocks/SM), and executed instructions exploded from ~5M to ~35M — a 6.9x increase — indicating the larger tile size caused excessive redundant computation or predicated-off work. The grid's waves_per_sm also jumped from 6.96 to 27.84, meaning the larger block size did not reduce grid size as expected (grid remained at 14,700), suggesting the launch site still used the fixed grid=cdiv(xnumel,128) rather than a lambda grid, forcing each block to process only a 128-element slice while paying 512-thread overhead.

> **Applicable when**: Never apply this specific pattern (autotune with lambda grid omitted). When adding autotune to a Triton kernel, the launch grid MUST be updated to a lambda that reads meta['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune without simultaneously converting the launch grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Leaving a fixed grid while removing the explicit XBLOCK kwarg causes the kernel to launch with the autotuned block size but the old (smaller) grid, resulting in massive thread waste and instruction explosion.

---

## Cumulative Progress

```
Baseline:  11810 ns  (1.000×)
Iter 1     11780 ns  (1.002×)  — autotune_xblock_for_occupancy_and_l1_reuse

Final best: 11780 ns  (1.0025×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
+], key=['xnumel'], restore_value=['in_out_ptr0'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1881600
@@ -33,7 +42,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -42,7 +50,7 @@

     arg_4 = inputs[4]
     xnumel = 1881600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
