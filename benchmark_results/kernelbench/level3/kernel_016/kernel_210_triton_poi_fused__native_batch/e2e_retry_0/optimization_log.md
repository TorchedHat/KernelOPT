# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_210` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6500 ns — **Best latency**: 6270 ns (**1.0367×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_210`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_210_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6003,1,1), Waves/SM: 2.84
- **Registers/thread**: 22
- **Duration**: 6500 ns
- **SM Throughput**: 40.92 % | **Memory**: 15.61 % | **DRAM**: 9.81 %
- **Achieved Occupancy**: 60.68 %
- **IPC Active**: 2.47 inst/cycle | **No Eligible**: 36.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.939× | 0.809× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.004× | 1.037× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6500 ns | 8030 ns | +1530 ns (+23.5%) | 6270 ns | -230 ns (-3.5%) |
| SM Throughput (%) | 40.92 % | 61.90 % | +20.98 pp | 31.53 % | -9.39 pp |
| Memory Throughput (%) | 15.61 % | 16.83 % | +1.22 pp | 21.85 % | +6.24 pp |
| DRAM Throughput (%) | 9.81 % | 7.88 % | -1.93 pp | 10.11 % | +0.30 pp |
| Achieved Occupancy (%) | 60.68 % | 78.54 % | +17.86 pp | 45.09 % | -15.59 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 23.00 register/thread | +1 |
| Waves/SM | 2.84 | 2.84 | +0.00 | 2.84 | +0.00 |
| No Eligible (%) | 36.11 % | 16.92 % | -19.19 pp | 49.26 % | +13.15 pp |
| IPC Active | 2.47 inst/cycle | 3.29 inst/cycle | +0.82 | 2.07 inst/cycle | -0.40 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_eliminate_partial_wave` (0.809×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the autotuner to select a configuration that increased registers per thread from 22 to 32, which reduced Block Limit Registers from 21 to 16 blocks/SM. Despite improving achieved occupancy (60.7% → 78.5%), the executed instruction count nearly doubled (2.04M → 3.85M) and kernel duration increased from 6.5 µs to 8.03 µs. The grid and wave count remained unchanged (6003 blocks, 2.84 waves/SM), indicating the autotune did not select a larger XBLOCK — instead the autotune overhead and/or the selected config introduced extra computation, making this a net regression. When the original kernel is already register-efficient (22 regs/thread) and the grid/wave structure does not change after autotuning, adding autotune overhead without actually reducing the grid size produces a slowdown.

> **Applicable when**: Never apply this pattern as implemented: adding @triton.autotune while keeping a fixed grid (not a lambda) means the grid does not adapt to the selected XBLOCK, so larger XBLOCK configs are never actually used to reduce the grid size.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs unless the grid launch is also changed to a lambda (e.g., `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`). Without this, the autotune overhead and register pressure increase with no wave-reduction benefit. Also avoid when the kernel is already register-efficient (≤24 regs/thread) and the autotune may select configs that push registers higher.

---

## Cumulative Progress

```
Baseline:  6500 ns  (1.000×)
Iter 1     6270 ns  (1.037×)  — autotune_xblock_eliminate_partial_wave

Final best: 6270 ns  (1.0367×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_210(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 768320
@@ -21,19 +33,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
+    tmp16 = tl.full([1], 0, tl.float32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +50,7 @@

     arg_5 = inputs[5]
     xnumel = 768320
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_210[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 768320, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_210[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 768320)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
