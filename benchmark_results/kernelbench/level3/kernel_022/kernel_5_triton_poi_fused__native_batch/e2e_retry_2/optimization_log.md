# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 66210 ns — **Best latency**: 55650 ns (**1.1898×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`software_pipelining_via_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 32
- **Duration**: 66210 ns
- **SM Throughput**: 26.15 % | **Memory**: 62.86 % | **DRAM**: 56.08 %
- **Achieved Occupancy**: 92.99 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 72.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | software_pipelining_via_autotune | 1.710× | 1.190× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | software_pipelini... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 66210 ns | 55650 ns | -10560 ns (-15.9%) |
| SM Throughput (%) | 26.15 % | 36.00 % | +9.85 pp |
| Memory Throughput (%) | 62.86 % | 84.62 % | +21.76 pp |
| DRAM Throughput (%) | 56.08 % | 28.21 % | -27.87 pp |
| Achieved Occupancy (%) | 92.99 % | 69.89 % | -23.10 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 |
| Waves/SM | 11.14 | 14.85 | +3.71 |
| No Eligible (%) | 72.89 % | 61.83 % | -11.06 pp |
| IPC Active | 1.11 inst/cycle | 1.51 inst/cycle | +0.40 |


### Key Learnings

#### ✅ Plan B: `software_pipelining_via_autotune` (1.190×)

When a Triton kernel shows >70% no-eligible-warp cycles and a dominant L1TEX scoreboard stall (>50% of CPI), the hardware cannot hide memory latency with the available active warps. Adding @triton.autotune with configs that sweep num_stages (2–4) alongside XBLOCK/YBLOCK/num_warps enables the Triton compiler to emit async prefetch instructions, overlapping the next tile's data fetch with current-tile compute. In this case, warp cycles per issued instruction dropped from 54.2 to 29.2 and eligible warps per scheduler rose from 0.57 to 1.0, yielding a 1.19x speedup (66.2 µs → 55.7 µs); the trade-off is higher register pressure (32→40 registers/thread) which reduced theoretical occupancy from 100% to 75%.

> **Applicable when**: Triton pointwise/elementwise kernel with: (1) no-eligible-warp cycles >60%, (2) dominant L1TEX scoreboard stall >40% of CPI, (3) L1 hit rate <10%, (4) current num_stages=1 (single-stage pipeline), (5) kernel body is simple (few loads, one FMA, one store) making it amenable to Triton's pipeline lowering.

> **Avoid when**: Kernel is already register-bound (registers/thread near hardware max); increasing num_stages raises register pressure and may reduce occupancy enough to hurt performance. Also avoid when the kernel has complex control flow or data-dependent branches that prevent static prefetch scheduling. Not applicable when the bottleneck is compute-bound (SM throughput >80%) rather than memory-latency-bound.

---

## Cumulative Progress

```
Baseline:  66210 ns  (1.000×)
Iter 1     55650 ns  (1.190×)  — software_pipelining_via_autotune

Final best: 55650 ns  (1.1898×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1920
@@ -37,7 +53,6 @@

     tmp15 = tmp0 * combined_scale + combined_bias
     tl.store(out_ptr0 + (x2 + 12544*y3), tmp15, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -48,7 +63,7 @@

     ynumel = 1920
     xnumel = 12544
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1920, 12544)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
