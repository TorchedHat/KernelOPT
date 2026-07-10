# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5920 ns — **Best latency**: 10080 ns (**0.5873×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5920 ns
- **SM Throughput**: 16.49 % | **Memory**: 42.68 % | **DRAM**: 20.78 %
- **Achieved Occupancy**: 54.82 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_coalesced_access_and_occupancy | 1.082× | 0.587× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5920 ns | 10080 ns | +4160 ns (+70.3%) |
| SM Throughput (%) | 16.49 % | 13.72 % | -2.77 pp |
| Memory Throughput (%) | 42.68 % | 31.96 % | -10.72 pp |
| DRAM Throughput (%) | 20.78 % | 6.12 % | -14.66 pp |
| Achieved Occupancy (%) | 54.82 % | 51.21 % | -3.61 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.56 % | 73.98 % | +2.42 pp |
| IPC Active | 1.16 inst/cycle | 1.04 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  5920 ns  (1.000×)

Final best: 10080 ns  (0.5873×)
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
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +33,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
