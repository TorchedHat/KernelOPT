# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 284900 ns — **Best latency**: 286050 ns (**0.9960×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 284900 ns
- **SM Throughput**: 19.67 % | **Memory**: 79.14 % | **DRAM**: 72.43 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_triton_autotune | 1.000× | 0.996× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 284900 ns | 286050 ns | +1150 ns (+0.4%) |
| SM Throughput (%) | 19.67 % | 19.57 % | -0.10 pp |
| Memory Throughput (%) | 79.14 % | 78.85 % | -0.29 pp |
| DRAM Throughput (%) | 72.43 % | 72.14 % | -0.29 pp |
| Achieved Occupancy (%) | 94.04 % | 94.03 % | -0.01 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.04 % | 80.34 % | +0.30 pp |
| IPC Active | 0.79 inst/cycle | 0.80 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  284900 ns  (1.000×)

Final best: 286050 ns  (0.9960×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 8}, num_warps=8),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
@@ -19,10 +26,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +35,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
