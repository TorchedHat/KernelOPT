# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5180 ns — **Best latency**: 4480 ns (**1.1562×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`increase_yblock_tile_size_and_autotune_to_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_11_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 28
- **Duration**: 5180 ns
- **SM Throughput**: 40.42 % | **Memory**: 39.86 % | **DRAM**: 4.65 %
- **Achieved Occupancy**: 68.59 %
- **IPC Active**: 2.71 inst/cycle | **No Eligible**: 29.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_yblock_tile_size_and_autotune_to_eliminate_partial_wave | — | 1.156× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_yblock_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5180 ns | 4480 ns | -700 ns (-13.5%) |
| SM Throughput (%) | 40.42 % | 7.34 % | -33.08 pp |
| Memory Throughput (%) | 39.86 % | 5.39 % | -34.47 pp |
| DRAM Throughput (%) | 4.65 % | 5.39 % | +0.74 pp |
| Achieved Occupancy (%) | 68.59 % | 27.52 % | -41.07 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 0.97 | -0.97 |
| No Eligible (%) | 29.63 % | 65.51 % | +35.88 pp |
| IPC Active | 2.71 inst/cycle | 1.02 inst/cycle | -1.69 |

---

## Cumulative Progress

```
Baseline:  5180 ns  (1.000×)
Iter 1     4480 ns  (1.156×)  — increase_yblock_tile_size_and_autotune_to_eliminate_partial_wave

Final best: 4480 ns  (1.1562×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'YBLOCK': 32,  'XBLOCK': 1},  num_warps=2),
+    triton.Config({'YBLOCK': 64,  'XBLOCK': 1},  num_warps=4),
+    triton.Config({'YBLOCK': 128, 'XBLOCK': 1},  num_warps=4),
+    triton.Config({'YBLOCK': 256, 'XBLOCK': 1},  num_warps=8),
+    triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2),
+    triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
+    triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 32768
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +29,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +38,7 @@

     ynumel = 32768
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
