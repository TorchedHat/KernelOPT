# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 6020 ns (**0.9884×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_056/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 15.88 % | **Memory**: 41.05 % | **DRAM**: 20.66 %
- **Achieved Occupancy**: 55.94 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 71.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.355× | 0.988× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5950 ns | 6020 ns | +70 ns (+1.2%) |
| SM Throughput (%) | 15.88 % | 14.58 % | -1.30 pp |
| Memory Throughput (%) | 41.05 % | 26.61 % | -14.44 pp |
| DRAM Throughput (%) | 20.66 % | 5.13 % | -15.53 pp |
| Achieved Occupancy (%) | 55.94 % | 41.42 % | -14.52 pp |
| Registers/Thread | 26.00 register/thread | 24.00 register/thread | -2 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.66 % | 67.60 % | -4.06 pp |
| IPC Active | 1.11 inst/cycle | 1.17 inst/cycle | +0.06 |

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)

Final best: 6020 ns  (0.9884×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
