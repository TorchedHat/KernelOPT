# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5220 ns — **Best latency**: 14690 ns (**0.3553×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_24_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 1.86
- **Registers/thread**: 18
- **Duration**: 5220 ns
- **SM Throughput**: 33.35 % | **Memory**: 12.03 % | **DRAM**: 7.94 %
- **Achieved Occupancy**: 62.00 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 38.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_autotune_eliminate_partial_wave | 0.972× | 0.978× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.002× | 0.355× | ✓ | **✓ best** |
| C | add_triton_autotune | 0.980× | 1.032× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_e... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5220 ns | 5340 ns | +120 ns (+2.3%) | 14690 ns | +9470 ns (+181.4%) | 5060 ns | -160 ns (-3.1%) |
| SM Throughput (%) | 33.35 % | 40.00 % | +6.65 pp | 75.25 % | +41.90 pp | 31.68 % | -1.67 pp |
| Memory Throughput (%) | 12.03 % | 13.92 % | +1.89 pp | 15.78 % | +3.75 pp | 11.41 % | -0.62 pp |
| DRAM Throughput (%) | 7.94 % | 7.77 % | -0.17 pp | 2.80 % | -5.14 pp | 8.15 % | +0.21 pp |
| Achieved Occupancy (%) | 62.00 % | 69.70 % | +7.70 pp | 53.54 % | -8.46 pp | 60.94 % | -1.06 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 48.00 register/thread | +30 | 18.00 register/thread | +0 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 5.94 | +4.08 | 1.86 | +0.00 |
| No Eligible (%) | 38.84 % | 21.80 % | -17.04 pp | 12.67 % | -26.17 pp | 37.91 % | -0.93 pp |
| IPC Active | 2.43 inst/cycle | 3.07 inst/cycle | +0.64 | 3.52 inst/cycle | +1.09 | 2.36 inst/cycle | -0.07 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.355×)

Adding @triton.autotune with large XBLOCK configs (256–2048) to a pointwise kernel caused the compiler to allocate significantly more registers per thread (18 → 48), which reduced theoretical occupancy from 100% to 62.5% (Block Limit Registers dropped from 21 to 5 blocks/SM) and increased kernel duration by ~2.8x. The autotune selected XBLOCK=256 with num_warps=4, which also increased total executed instructions by 6.3x, indicating the larger tile size introduced redundant computation or loop overhead that overwhelmed any partial-wave benefit. When a pointwise kernel already has low register pressure at the baseline block size, adding autotune configs with larger XBLOCK can trigger register pressure increases that cap occupancy and negate any launch configuration gains.

> **Applicable when**: Kernel has very low register pressure at baseline (≤20 registers/thread), theoretical occupancy is already at or near 100%, and the partial-wave penalty is the primary NCU complaint. The kernel must be truly register-light so that larger XBLOCK does not trigger register spill.

> **Avoid when**: The baseline kernel already uses ≥18 registers/thread in a pointwise pattern; adding autotune with larger XBLOCK configs will cause the compiler to allocate more registers (observed: 18→48), reducing Block Limit Registers and capping theoretical occupancy below the baseline. Do not apply when the instruction count increase from larger tiles (6x observed) would outweigh partial-wave savings. Avoid when the kernel is already compute-bound at baseline (SM Throughput >60%).

---

## Cumulative Progress

```
Baseline:  5220 ns  (1.000×)

Final best: 14690 ns  (0.3553×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 501760
@@ -33,8 +46,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -43,7 +54,7 @@

     arg_4 = inputs[4]
     xnumel = 501760
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
