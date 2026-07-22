# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2980 ns — **Best latency**: 2820 ns (**1.0567×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_and_autotune_for_small_grid_utilization`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_10_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (640,1,1), Waves/SM: 0.30
- **Registers/thread**: 28
- **Duration**: 2980 ns
- **SM Throughput**: 11.33 % | **Memory**: 10.34 % | **DRAM**: 1.30 %
- **Achieved Occupancy**: 25.36 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 64.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_grid_utilization | 1.047× | 1.096× | ✓ |  |
| B | fix_tile_sizes_and_autotune_for_small_grid_utilization | 1.057× | 1.057× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2980 ns | 2720 ns | -260 ns (-8.7%) | 2820 ns | -160 ns (-5.4%) |
| SM Throughput (%) | 11.33 % | 6.94 % | -4.39 pp | 4.58 % | -6.75 pp |
| Memory Throughput (%) | 10.34 % | 6.25 % | -4.09 pp | 3.83 % | -6.51 pp |
| DRAM Throughput (%) | 1.30 % | 1.48 % | +0.18 pp | 1.38 % | +0.08 pp |
| Achieved Occupancy (%) | 25.36 % | 10.81 % | -14.55 pp | 10.82 % | -14.54 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 21.00 register/thread | -7 |
| Waves/SM | 0.30 | 0.15 | -0.15 | 0.15 | -0.15 |
| No Eligible (%) | 64.39 % | 70.77 % | +6.38 pp | 86.34 % | +21.95 pp |
| IPC Active | 1.32 inst/cycle | 0.93 inst/cycle | -0.39 | 0.58 inst/cycle | -0.74 |

---

## Cumulative Progress

```
Baseline:  2980 ns  (1.000×)
Iter 1     2820 ns  (1.057×)  — fix_tile_sizes_and_autotune_for_small_grid_utilization

Final best: 2820 ns  (1.0567×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 5120
@@ -22,14 +33,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, mask=xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 5120
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10[grid](arg_0, arg_1, 5120, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10[grid](arg_0, arg_1, 5120, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
