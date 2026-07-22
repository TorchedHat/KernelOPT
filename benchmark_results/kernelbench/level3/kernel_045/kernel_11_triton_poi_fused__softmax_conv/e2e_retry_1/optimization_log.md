# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5180 ns — **Best latency**: 5090 ns (**1.0177×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_eliminate_partial_wave`** beats baseline latency.

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
- **SM Throughput**: 40.38 % | **Memory**: 39.87 % | **DRAM**: 4.66 %
- **Achieved Occupancy**: 71.01 %
- **IPC Active**: 2.79 inst/cycle | **No Eligible**: 30.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_sizes_eliminate_partial_wave | — | 1.018× | ✓ | **✓ best** |
| C | add_triton_autotune | — | 1.149× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_el... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 5090 ns | -90 ns (-1.7%) | 4510 ns | -670 ns (-12.9%) |
| SM Throughput (%) | 40.38 % | 23.83 % | -16.55 pp | 7.32 % | -33.06 pp |
| Memory Throughput (%) | 39.87 % | 22.66 % | -17.21 pp | 5.37 % | -34.50 pp |
| DRAM Throughput (%) | 4.66 % | 4.77 % | +0.11 pp | 5.37 % | +0.71 pp |
| Achieved Occupancy (%) | 71.01 % | 62.61 % | -8.40 pp | 33.68 % | -37.33 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 0.97 | -0.97 |
| No Eligible (%) | 30.04 % | 33.85 % | +3.81 pp | 66.16 % | +36.12 pp |
| IPC Active | 2.79 inst/cycle | 2.52 inst/cycle | -0.27 | 1.00 inst/cycle | -1.79 |

---

## Cumulative Progress

```
Baseline:  5180 ns  (1.000×)
Iter 1     5090 ns  (1.018×)  — fix_tile_sizes_eliminate_partial_wave

Final best: 5090 ns  (1.0177×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=2),
+    triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+    triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+    triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
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
@@ -19,10 +31,8 @@

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
@@ -30,7 +40,7 @@

     ynumel = 32768
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
