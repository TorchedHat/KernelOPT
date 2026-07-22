# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_224` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6880 ns — **Best latency**: 6880 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_224`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_224_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6860,1,1), Waves/SM: 3.25
- **Registers/thread**: 22
- **Duration**: 6880 ns
- **SM Throughput**: 42.98 % | **Memory**: 16.34 % | **DRAM**: 10.54 %
- **Achieved Occupancy**: 59.10 %
- **IPC Active**: 2.46 inst/cycle | **No Eligible**: 35.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_reduce_partial_wave_and_improve_occupancy | 1.001× | 1.000× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 0.926× | 0.776× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6880 ns | 6880 ns | +0 ns (+0.0%) | 8860 ns | +1980 ns (+28.8%) |
| SM Throughput (%) | 42.98 % | 42.31 % | -0.67 pp | 63.23 % | +20.25 pp |
| Memory Throughput (%) | 16.34 % | 16.10 % | -0.24 pp | 17.06 % | +0.72 pp |
| DRAM Throughput (%) | 10.54 % | 10.56 % | +0.02 pp | 8.18 % | -2.36 pp |
| Achieved Occupancy (%) | 59.10 % | 63.64 % | +4.54 pp | 80.51 % | +21.41 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 |
| Waves/SM | 3.25 | 3.25 | +0.00 | 3.25 | +0.00 |
| No Eligible (%) | 35.11 % | 35.77 % | +0.66 pp | 17.90 % | -17.21 pp |
| IPC Active | 2.46 inst/cycle | 2.56 inst/cycle | +0.10 | 3.27 inst/cycle | +0.81 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_eliminate_partial_wave` (0.776×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (22→32), which nearly doubled the executed instruction count (2.33M→4.42M) and increased kernel duration by ~29% despite improving achieved occupancy (59%→80%). The root cause is that larger tile sizes force the compiler to keep more live values in registers, increasing register pressure and instruction count faster than the latency-hiding benefit of higher occupancy. When the kernel is already short (< 10 µs) and the instruction count increase outweighs the occupancy gain, autotuning larger blocks is counterproductive.

> **Applicable when**: Kernel is memory-bandwidth bound with low register pressure (< 24 registers/thread), and the primary bottleneck is genuinely partial-wave tail latency with no instruction-count sensitivity to tile size.

> **Avoid when**: Kernel is short-running (< 10 µs), register count is already near 22–24 (leaving little headroom), and the instruction count is sensitive to XBLOCK size. In these cases, larger XBLOCK inflates register pressure and instruction count, causing regression even when occupancy improves. Also avoid when the kernel is compute-bound (SM throughput > 60%) rather than memory-bound.

---

## Cumulative Progress

```
Baseline:  6880 ns  (1.000×)

Final best: 6880 ns  (1.0000×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_224(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 878080
@@ -33,8 +43,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +52,7 @@

     arg_5 = inputs[5]
     xnumel = 878080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_224[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 878080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_224[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 878080)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
