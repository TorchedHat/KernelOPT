# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4420 ns — **Best latency**: 4480 ns (**0.9866×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_52_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2940,1,1), Waves/SM: 1.39
- **Registers/thread**: 18
- **Duration**: 4420 ns
- **SM Throughput**: 25.95 % | **Memory**: 11.25 % | **DRAM**: 7.07 %
- **Achieved Occupancy**: 63.35 %
- **IPC Active**: 2.13 inst/cycle | **No Eligible**: 45.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_autotune_xblock | 1.027× | 0.705× | ✓ |  |
| B | add_autotune_increase_xblock_eliminate_partial_wave | 1.029× | 0.987× | ✓ | **✓ best** |
| C | add_triton_autotune | 0.993× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | add_autotune_incr... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4420 ns | 6270 ns | +1850 ns (+41.9%) | 4480 ns | +60 ns (+1.4%) | 4540 ns | +120 ns (+2.7%) |
| SM Throughput (%) | 25.95 % | 56.42 % | +30.47 pp | 39.53 % | +13.58 pp | 25.85 % | -0.10 pp |
| Memory Throughput (%) | 11.25 % | 26.41 % | +15.16 pp | 21.35 % | +10.10 pp | 10.84 % | -0.41 pp |
| DRAM Throughput (%) | 7.07 % | 4.93 % | -2.14 pp | 6.96 % | -0.11 pp | 6.82 % | -0.25 pp |
| Achieved Occupancy (%) | 63.35 % | 77.09 % | +13.74 pp | 69.60 % | +6.25 pp | 64.68 % | +1.33 pp |
| Registers/Thread | 18.00 register/thread | 31.00 register/thread | +13 | 31.00 register/thread | +13 | 18.00 register/thread | +0 |
| Waves/SM | 1.39 | 2.78 | +1.39 | 1.39 | +0.00 | 1.39 | +0.00 |
| No Eligible (%) | 45.84 % | 17.46 % | -28.38 pp | 20.98 % | -24.86 pp | 45.22 % | -0.62 pp |
| IPC Active | 2.13 inst/cycle | 3.28 inst/cycle | +1.15 | 2.82 inst/cycle | +0.69 | 1.95 inst/cycle | -0.18 |


### Key Learnings

#### ❌ Plan A: `eliminate_partial_wave_via_autotune_xblock` (0.705×)

Adding @triton.autotune with larger XBLOCK configs (256, 512, 1024) to a short-duration pointwise kernel caused a 42% regression (4.42 us → 6.27 us). Although occupancy improved (63% → 77%) and scheduler utilization rose (54% → 82%), the autotuner selected XBLOCK=256 which doubled the register pressure (18 → 31 registers/thread), tripling total executed instructions (917K → 2.85M) and nearly doubling elapsed cycles. The LaunchConfiguration partial-wave penalty (~50% NCU estimate) was not the true bottleneck for this kernel; the kernel was already fast enough that autotune overhead and register spill from larger blocks dominated. For very short kernels (<5 us) with low register counts, adding autotune with larger block sizes can increase register pressure and instruction count enough to negate any wave-tail savings.

> **Applicable when**: Kernel has many waves with a large partial-wave tail (waves_per_sm >> 2), high register headroom (registers_per_thread << 32), and long duration (>10 us) where wave-tail overhead is a meaningful fraction of runtime.

> **Avoid when**: Kernel duration is short (<5 us); registers_per_thread is already moderate (>15) and larger XBLOCK would push toward register spill; the grid was not updated to use meta['XBLOCK'] (causing the autotuner to always use the fixed grid size regardless of selected config); the kernel is already near the partial-wave boundary with only 1-2 full waves.

---

## Cumulative Progress

```
Baseline:  4420 ns  (1.000×)

Final best: 4480 ns  (0.9866×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 376320
@@ -31,8 +42,6 @@

     tmp13 = tmp11 + tmp12
     tl.store(out_ptr0 + (x3), tmp13, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -42,7 +51,7 @@

     arg_5 = inputs[5]
     xnumel = 376320
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 376320, XBLOCK=128)
+    triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 376320)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
