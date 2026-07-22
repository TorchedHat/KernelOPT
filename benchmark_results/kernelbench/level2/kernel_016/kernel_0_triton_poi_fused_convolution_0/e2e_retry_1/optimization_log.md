# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 6240 ns (**0.9279×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 17.12 % | **Memory**: 42.50 % | **DRAM**: 21.28 %
- **Achieved Occupancy**: 54.64 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 73.00 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 0.980× | 0.978× | ✓ |  |
| B | fix_uncoalesced_stores_via_autotune_tile_reorder | 1.208× | 0.928× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.003× | 0.943× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 5920 ns | +130 ns (+2.2%) | 6240 ns | +450 ns (+7.8%) | 6140 ns | +350 ns (+6.0%) |
| SM Throughput (%) | 17.12 % | 16.66 % | -0.46 pp | 15.04 % | -2.08 pp | 17.45 % | +0.33 pp |
| Memory Throughput (%) | 42.50 % | 41.56 % | -0.94 pp | 27.43 % | -15.07 pp | 40.10 % | -2.40 pp |
| DRAM Throughput (%) | 21.28 % | 20.87 % | -0.41 pp | 4.96 % | -16.32 pp | 20.08 % | -1.20 pp |
| Achieved Occupancy (%) | 54.64 % | 54.78 % | +0.14 pp | 41.35 % | -13.29 pp | 54.62 % | -0.02 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 24.00 register/thread | -2 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 73.00 % | 71.15 % | -1.85 pp | 68.05 % | -4.95 pp | 70.65 % | -2.35 pp |
| IPC Active | 1.16 inst/cycle | 1.16 inst/cycle | +0.00 | 1.13 inst/cycle | -0.03 | 1.16 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)

Final best: 6240 ns  (0.9279×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +34,13 @@

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
